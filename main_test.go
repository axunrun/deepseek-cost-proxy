package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNormalizeRequestAddsDefaultModelAndStreamUsage(t *testing.T) {
	body := []byte(`{"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	normalized, err := normalizeRequest(body, "deepseek-v4-flash")
	if err != nil {
		t.Fatalf("normalizeRequest: %v", err)
	}
	if normalized.model != "deepseek-v4-flash" {
		t.Fatalf("model = %q", normalized.model)
	}
	if !normalized.stream {
		t.Fatal("stream should be true")
	}
	if !strings.Contains(string(normalized.body), `"include_usage":true`) {
		t.Fatalf("stream_options.include_usage missing: %s", normalized.body)
	}
}

func TestNormalizeRequestRejectsUnsupportedModel(t *testing.T) {
	_, err := normalizeRequest([]byte(`{"model":"other"}`), "deepseek-v4-flash")
	if err == nil {
		t.Fatal("expected unsupported model error")
	}
}

func TestNormalizeRequestSortsToolsAndRecordsDebugTrace(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"messages":[{"role":"system","content":"stable"},{"role":"user","content":"hi"}],
		"tools":[
			{"type":"function","function":{"name":"zeta","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"alpha","parameters":{"type":"object"}}}
		]
	}`)

	normalized, err := normalizeRequest(body, "deepseek-v4-flash")
	if err != nil {
		t.Fatalf("normalizeRequest: %v", err)
	}
	if got := strings.Index(string(normalized.body), `"name":"alpha"`); got < 0 {
		t.Fatalf("normalized body missing alpha: %s", normalized.body)
	}
	if strings.Index(string(normalized.body), `"name":"alpha"`) >
		strings.Index(string(normalized.body), `"name":"zeta"`) {
		t.Fatalf("tools were not sorted: %s", normalized.body)
	}
	if !normalized.trace.ToolsChanged {
		t.Fatalf("trace should record tool change: %+v", normalized.trace)
	}
	if normalized.trace.SystemChanged {
		t.Fatalf("system should not change: %+v", normalized.trace)
	}
	if normalized.trace.RawToolsOrder[0] != "zeta" ||
		normalized.trace.NormToolsOrder[0] != "alpha" {
		t.Fatalf("tool orders not captured: raw=%v normalized=%v",
			normalized.trace.RawToolsOrder,
			normalized.trace.NormToolsOrder,
		)
	}
}

func TestCopyStreamAndCaptureUsage(t *testing.T) {
	input := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"OK"}}],"usage":null}`,
		``,
		`data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_cache_hit_tokens":80,"prompt_cache_miss_tokens":20}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var out bytes.Buffer

	metric, err := copyStreamAndCaptureUsage(&out, strings.NewReader(input), "deepseek-v4-flash", 200)
	if err != nil {
		t.Fatalf("copyStreamAndCaptureUsage: %v", err)
	}
	if !strings.Contains(out.String(), "[DONE]") {
		t.Fatalf("stream was not forwarded: %q", out.String())
	}
	if metric.PromptTokens != 100 ||
		metric.PromptCacheHitTokens != 80 ||
		metric.PromptCacheMissTokens != 20 ||
		metric.CompletionTokens != 5 ||
		metric.TotalTokens != 105 {
		t.Fatalf("metric = %+v", metric)
	}
	if metric.HitRate != 0.8 {
		t.Fatalf("hit rate = %v", metric.HitRate)
	}
}

func TestMetricsStorePersistsAndLoadsRequests(t *testing.T) {
	dir := t.TempDir()
	store := newMetricsStore(dir)

	metric := requestMetric{
		Time:                  storeTimeForTest(),
		Model:                 "deepseek-v4-flash",
		Status:                200,
		PromptTokens:          100,
		PromptCacheHitTokens:  80,
		PromptCacheMissTokens: 20,
		HitRate:               0.8,
	}
	trace := debugTrace{
		Time:              metric.Time,
		Model:             metric.Model,
		RawPrefixHash:     "raw",
		NormPrefixHash:    "norm",
		RawPreview:        []byte(`{"model":"deepseek-v4-flash"}`),
		NormalizedPreview: []byte(`{"model":"deepseek-v4-flash"}`),
	}

	store.add(metric, trace)
	reloaded := newMetricsStore(dir)
	snapshot := reloaded.snapshot()
	summary := snapshot["summary"].(map[string]any)

	if summary["requests"] != 1 {
		t.Fatalf("requests = %v", summary["requests"])
	}
	loadedTrace, ok := reloaded.debugByID(1)
	if !ok {
		t.Fatal("debug trace was not loaded")
	}
	if loadedTrace.RawPrefixHash != "raw" || loadedTrace.NormPrefixHash != "norm" {
		t.Fatalf("loaded trace = %+v", loadedTrace)
	}
}

func TestHandleChatCompletionsPersistsTrace(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"test",
			"choices":[{"message":{"role":"assistant","content":"OK"}}],
			"usage":{
				"prompt_tokens":100,
				"completion_tokens":5,
				"total_tokens":105,
				"prompt_cache_hit_tokens":80,
				"prompt_cache_miss_tokens":20
			}
		}`))
	}))
	defer upstream.Close()

	traceDir := t.TempDir()
	cfg := config{
		deepSeekKey:  "sk-test",
		proxyAuthKey: "local-proxy-key",
		defaultModel: "deepseek-v4-flash",
		deepSeekURL:  upstream.URL,
		traceDir:     traceDir,
	}
	metrics := newMetricsStore(traceDir)
	body := `{
		"model":"deepseek-v4-flash",
		"messages":[{"role":"system","content":"stable"},{"role":"user","content":"hi"}],
		"tools":[
			{"type":"function","function":{"name":"zeta"}},
			{"type":"function","function":{"name":"alpha"}}
		]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer local-proxy-key")
	rec := httptest.NewRecorder()

	handleChatCompletions(cfg, metrics, rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	snapshot := metrics.snapshot()
	summary := snapshot["summary"].(map[string]any)
	if summary["cachedTokens"] != 80 {
		t.Fatalf("summary = %+v", summary)
	}
	trace, ok := metrics.debugByID(1)
	if !ok {
		t.Fatal("debug trace missing")
	}
	if !trace.ToolsChanged {
		t.Fatalf("expected toolsChanged trace: %+v", trace)
	}
	data, err := os.ReadFile(filepath.Join(traceDir, "requests.jsonl"))
	if err != nil {
		t.Fatalf("read persisted trace: %v", err)
	}
	var persisted persistedRequest
	if err := json.Unmarshal(bytes.TrimSpace(data), &persisted); err != nil {
		t.Fatalf("parse persisted trace: %v", err)
	}
	if persisted.Metric.PromptCacheHitTokens != 80 || persisted.Trace.ID != 1 {
		t.Fatalf("persisted = %+v", persisted)
	}
}

func storeTimeForTest() time.Time {
	return time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
}
