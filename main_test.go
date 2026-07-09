package main

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testSupportedModels() map[string]bool {
	return map[string]bool{
		"deepseek-v4-flash": true,
		"deepseek-v4-pro":   true,
		"MiniMax-M3":        true,
	}
}

func testConfig(deepSeekKey, miniMaxKey string) config {
	cfg := config{
		defaultModel: "deepseek-v4-flash",
		models:       map[string]modelConfig{},
		supported:    testSupportedModels(),
	}
	if deepSeekKey != "" {
		cfg.addModel("deepseek-v4-flash", "deepseek", "https://deepseek.test/chat", deepSeekKey)
		cfg.addModel("deepseek-v4-pro", "deepseek", "https://deepseek.test/chat", deepSeekKey)
	}
	if miniMaxKey != "" {
		cfg.addModel("MiniMax-M3", "minimax", "https://minimax.test/chat", miniMaxKey)
	}
	return cfg
}

func TestNormalizeRequestAddsDefaultModelAndStreamUsage(t *testing.T) {
	body := []byte(`{"stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	normalized, err := normalizeRequest(body, "deepseek-v4-flash", testSupportedModels())
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
	_, err := normalizeRequest([]byte(`{"model":"other"}`), "deepseek-v4-flash", testSupportedModels())
	if err == nil {
		t.Fatal("expected unsupported model error")
	}
}

func TestListenAddrAcceptsPortOnly(t *testing.T) {
	tests := map[string]string{
		"":                ":18188",
		"18188":           ":18188",
		":18188":          ":18188",
		"127.0.0.1:18188": "127.0.0.1:18188",
	}
	for input, want := range tests {
		if got := listenAddr(input); got != want {
			t.Fatalf("listenAddr(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestHandleModelsReturnsOpenAIModelList(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	handleModels(testConfig("sk-test", ""), rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"id":"deepseek-v4-flash"`) ||
		!strings.Contains(rec.Body.String(), `"id":"deepseek-v4-pro"`) ||
		strings.Contains(rec.Body.String(), `"id":"MiniMax-M3"`) {
		t.Fatalf("models response = %s", rec.Body.String())
	}
}

func TestHandleModelsIncludesConfiguredMiniMax(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	handleModels(testConfig("sk-test", "sk-mini"), rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"id":"MiniMax-M3"`) ||
		!strings.Contains(rec.Body.String(), `"owned_by":"minimax"`) {
		t.Fatalf("models response = %s", rec.Body.String())
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

	normalized, err := normalizeRequest(body, "deepseek-v4-flash", testSupportedModels())
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

func TestApplyUsageUsesModelPricing(t *testing.T) {
	u := usage{
		PromptTokens:          2_000_000,
		PromptCacheHitTokens:  1_000_000,
		PromptCacheMissTokens: 1_000_000,
		CompletionTokens:      1_000_000,
		TotalTokens:           3_000_000,
	}
	flash := requestMetric{Model: "deepseek-v4-flash"}
	pro := requestMetric{Model: "deepseek-v4-pro"}

	applyUsage(&flash, u)
	applyUsage(&pro, u)

	if flash.EstimatedCostCNY != 3.02 || flash.EstimatedSavedCNY != 0.98 {
		t.Fatalf("flash cost = %+v", flash)
	}
	if pro.EstimatedCostCNY != 9.025 || pro.EstimatedSavedCNY != 2.975 {
		t.Fatalf("pro cost = %+v", pro)
	}
}

func TestApplyUsageReadsOpenAIStyleCachedTokens(t *testing.T) {
	u := usage{
		PromptTokens:     100,
		CompletionTokens: 5,
		TotalTokens:      105,
		PromptTokensDetails: &tokenDetails{
			CachedTokens: 80,
		},
	}
	metric := requestMetric{Model: "MiniMax-M3"}

	applyUsage(&metric, u)

	if metric.PromptCacheHitTokens != 80 || metric.PromptCacheMissTokens != 20 {
		t.Fatalf("metric = %+v", metric)
	}
	if metric.Currency != "CNY" {
		t.Fatalf("currency = %q", metric.Currency)
	}
	if math.Abs(metric.EstimatedCost-0.0001176) > 0.000000001 ||
		math.Abs(metric.EstimatedSaved-0.0001344) > 0.000000001 {
		t.Fatalf("cost = %+v", metric)
	}
	if metric.EstimatedCostCNY != metric.EstimatedCost ||
		metric.EstimatedSavedCNY != metric.EstimatedSaved {
		t.Fatalf("MiniMax should be folded into CNY totals: %+v", metric)
	}
}

func TestPrefixChangeReasons(t *testing.T) {
	prevMetric := requestMetric{Model: "deepseek-v4-flash"}
	curMetric := requestMetric{Model: "deepseek-v4-pro"}
	prevTrace := debugTrace{
		NormSystemHash:   "system-a",
		NormToolsHash:    "tools-a",
		NormThinkingHash: "thinking-a",
		NormPrefixHash:   "prefix-a",
	}
	curTrace := debugTrace{
		NormSystemHash:   "system-b",
		NormToolsHash:    "tools-b",
		NormThinkingHash: "thinking-b",
		NormPrefixHash:   "prefix-b",
	}

	reasons := prefixChangeReasons(prevMetric, prevTrace, curMetric, curTrace)

	want := []string{"model", "system", "tools", "thinking"}
	if strings.Join(reasons, ",") != strings.Join(want, ",") {
		t.Fatalf("reasons = %v, want %v", reasons, want)
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

func TestStorageStatusReportsWritableTraceDir(t *testing.T) {
	store := newMetricsStore(filepath.Join(t.TempDir(), "traces"))

	status := store.storageStatus()

	if status["writable"] != true {
		t.Fatalf("storage status = %+v", status)
	}
	if status["traceDir"] == "" || status["path"] == "" {
		t.Fatalf("storage status missing paths: %+v", status)
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
	cfg := testConfig("sk-test", "")
	cfg.proxyAuthKey = "local-proxy-key"
	cfg.models["deepseek-v4-flash"] = modelConfig{
		ID:       "deepseek-v4-flash",
		Provider: "deepseek",
		ChatURL:  upstream.URL,
		APIKey:   "sk-test",
	}
	cfg.traceDir = traceDir
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

func TestHandleChatCompletionsRoutesMiniMaxModel(t *testing.T) {
	var upstreamAuth string
	var upstreamBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamAuth = r.Header.Get("Authorization")
		data, _ := io.ReadAll(r.Body)
		upstreamBody = string(data)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"mini",
			"choices":[{"message":{"role":"assistant","content":"OK"}}],
			"usage":{
				"prompt_tokens":100,
				"completion_tokens":5,
				"total_tokens":105,
				"prompt_tokens_details":{"cached_tokens":80}
			}
		}`))
	}))
	defer upstream.Close()

	cfg := testConfig("", "sk-mini")
	cfg.defaultModel = "MiniMax-M3"
	cfg.proxyAuthKey = "local-proxy-key"
	cfg.models["MiniMax-M3"] = modelConfig{
		ID:       "MiniMax-M3",
		Provider: "minimax",
		ChatURL:  upstream.URL,
		APIKey:   "sk-mini",
	}
	metrics := newMetricsStore(t.TempDir())
	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"MiniMax-M3","messages":[{"role":"user","content":"hi"}]}`),
	)
	req.Header.Set("Authorization", "Bearer local-proxy-key")
	rec := httptest.NewRecorder()

	handleChatCompletions(cfg, metrics, rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if upstreamAuth != "Bearer sk-mini" {
		t.Fatalf("upstream auth = %q", upstreamAuth)
	}
	if !strings.Contains(upstreamBody, `"model":"MiniMax-M3"`) {
		t.Fatalf("upstream body = %s", upstreamBody)
	}
	summary := metrics.snapshot()["summary"].(map[string]any)
	costByCurrency := summary["costByCurrency"].(map[string]float64)
	if costByCurrency["CNY"] == 0 {
		t.Fatalf("summary = %+v", summary)
	}
}

func storeTimeForTest() time.Time {
	return time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
}
