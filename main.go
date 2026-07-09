package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	defaultPort        = "18188"
	defaultDeepSeekURL = "https://api.deepseek.com/chat/completions"
)

var allowedModels = map[string]bool{
	"deepseek-v4-flash": true,
	"deepseek-v4-pro":   true,
}

type config struct {
	addr         string
	deepSeekKey  string
	proxyAuthKey string
	defaultModel string
	deepSeekURL  string
	traceDir     string
}

type chatRequest struct {
	Model string `json:"model"`
}

type usage struct {
	PromptTokens          int `json:"prompt_tokens"`
	CompletionTokens      int `json:"completion_tokens"`
	TotalTokens           int `json:"total_tokens"`
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
}

type bufferedResponse struct {
	Usage usage `json:"usage"`
}

type streamChunk struct {
	Usage *usage `json:"usage"`
}

type metricsStore struct {
	mu       sync.Mutex
	nextID   int
	requests []requestMetric
	debug    []debugTrace
	traceDir string
}

type persistedRequest struct {
	Metric requestMetric `json:"metric"`
	Trace  debugTrace    `json:"trace"`
}

type requestMetric struct {
	ID                    int       `json:"id"`
	Time                  time.Time `json:"time"`
	Model                 string    `json:"model"`
	Stream                bool      `json:"stream"`
	Status                int       `json:"status"`
	PromptTokens          int       `json:"promptTokens"`
	CompletionTokens      int       `json:"completionTokens"`
	TotalTokens           int       `json:"totalTokens"`
	PromptCacheHitTokens  int       `json:"promptCacheHitTokens"`
	PromptCacheMissTokens int       `json:"promptCacheMissTokens"`
	HitRate               float64   `json:"hitRate"`
	RawPrefixHash         string    `json:"rawPrefixHash,omitempty"`
	NormalizedPrefixHash  string    `json:"normalizedPrefixHash,omitempty"`
	ToolsChanged          bool      `json:"toolsChanged"`
	SystemChanged         bool      `json:"systemChanged"`
	EstimatedCostCNY      float64   `json:"estimatedCostCNY"`
	EstimatedSavedCNY     float64   `json:"estimatedSavedCNY"`
}

type debugTrace struct {
	ID                int             `json:"id"`
	Time              time.Time       `json:"time"`
	Model             string          `json:"model"`
	Stream            bool            `json:"stream"`
	RawPrefixHash     string          `json:"rawPrefixHash"`
	NormPrefixHash    string          `json:"normalizedPrefixHash"`
	RawSystemHash     string          `json:"rawSystemHash"`
	NormSystemHash    string          `json:"normalizedSystemHash"`
	RawToolsHash      string          `json:"rawToolsHash"`
	NormToolsHash     string          `json:"normalizedToolsHash"`
	ToolsChanged      bool            `json:"toolsChanged"`
	SystemChanged     bool            `json:"systemChanged"`
	RawToolsOrder     []string        `json:"rawToolsOrder,omitempty"`
	NormToolsOrder    []string        `json:"normalizedToolsOrder,omitempty"`
	RawPreview        json.RawMessage `json:"rawPreview"`
	NormalizedPreview json.RawMessage `json:"normalizedPreview"`
}

type normalizedRequest struct {
	body   []byte
	model  string
	stream bool
	trace  debugTrace
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	metrics := newMetricsStore(cfg.traceDir)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1", handleRoot)
	mux.HandleFunc("/v1/", handleRoot)
	mux.HandleFunc("/healthz", handleHealth)
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		handleMetrics(metrics, w, r)
	})
	mux.HandleFunc("/debug/requests", func(w http.ResponseWriter, r *http.Request) {
		handleDebugRequests(metrics, w, r)
	})
	mux.HandleFunc("/debug/requests/", func(w http.ResponseWriter, r *http.Request) {
		handleDebugRequest(metrics, w, r)
	})
	mux.HandleFunc("/debug", func(w http.ResponseWriter, r *http.Request) {
		handleDebugPage(w, r)
	})
	mux.HandleFunc("/dashboard", func(w http.ResponseWriter, r *http.Request) {
		handleDashboard(w, r)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		handleChatCompletions(cfg, metrics, w, r)
	})

	server := &http.Server{
		Addr:              cfg.addr,
		Handler:           mux,
		ReadHeaderTimeout: 15 * time.Second,
	}

	log.Printf("deepseek-cost-proxy listening on %s", cfg.addr)
	log.Fatal(server.ListenAndServe())
}

func loadConfig() (config, error) {
	cfg := config{
		addr:         listenAddr(envOrDefault("PROXY_ADDR", defaultPort)),
		proxyAuthKey: os.Getenv("PROXY_AUTH_KEY"),
		defaultModel: envOrDefault("DEFAULT_MODEL", "deepseek-v4-flash"),
		deepSeekURL:  envOrDefault("DEEPSEEK_CHAT_URL", defaultDeepSeekURL),
		deepSeekKey:  os.Getenv("DEEPSEEK_API_KEY"),
		traceDir:     envOrDefault("TRACE_DIR", "traces"),
	}
	if cfg.deepSeekKey == "" {
		return cfg, errors.New("DEEPSEEK_API_KEY is required")
	}
	if !allowedModels[cfg.defaultModel] {
		return cfg, fmt.Errorf("DEFAULT_MODEL %q is not supported", cfg.defaultModel)
	}
	return cfg, nil
}

func listenAddr(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ":" + defaultPort
	}
	if strings.Contains(value, ":") {
		return value
	}
	return ":" + value
}

func envOrDefault(name, fallback string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	return value
}

func handleRoot(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"name":   "deepseek-cost-proxy",
		"models": []string{"deepseek-v4-flash", "deepseek-v4-pro"},
	})
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func handleChatCompletions(cfg config, metrics *metricsStore, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !authorized(cfg, r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read request: "+err.Error(), http.StatusBadRequest)
		return
	}

	normalized, err := normalizeRequest(body, cfg.defaultModel)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, cfg.deepSeekURL, bytes.NewReader(normalized.body))
	if err != nil {
		http.Error(w, "build upstream request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.deepSeekKey)
	if accept := r.Header.Get("Accept"); accept != "" {
		req.Header.Set("Accept", accept)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		http.Error(w, "deepseek request: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if isStreamResponse(resp.Header) {
		metric, err := copyStreamAndCaptureUsage(w, resp.Body, normalized.model, resp.StatusCode)
		attachTrace(&metric, normalized.trace)
		if err != nil {
			log.Printf("model=%s stream=true copy failed: %v", normalized.model, err)
		}
		metrics.add(metric, normalized.trace)
		logUsage(metric)
		return
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("model=%s read upstream response failed: %v", normalized.model, err)
		return
	}
	metric := metricFromResponse(normalized.model, normalized.stream, resp.StatusCode, respBody)
	attachTrace(&metric, normalized.trace)
	metrics.add(metric, normalized.trace)
	logUsage(metric)
	_, _ = w.Write(respBody)
}

func authorized(cfg config, r *http.Request) bool {
	if cfg.proxyAuthKey == "" {
		return true
	}
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	return auth == "Bearer "+cfg.proxyAuthKey
}

func normalizeRequest(body []byte, defaultModel string) (normalizedRequest, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return normalizedRequest{}, fmt.Errorf("invalid JSON: %w", err)
	}
	rawForTrace := cloneMap(raw)

	model, _ := raw["model"].(string)
	model = strings.TrimSpace(model)
	if model == "" {
		model = defaultModel
		raw["model"] = model
	}
	if !allowedModels[model] {
		return normalizedRequest{}, fmt.Errorf("unsupported model %q; use deepseek-v4-flash or deepseek-v4-pro", model)
	}

	stream, _ := raw["stream"].(bool)
	sortTools(raw)
	if stream {
		opts, _ := raw["stream_options"].(map[string]any)
		if opts == nil {
			opts = map[string]any{}
		}
		opts["include_usage"] = true
		raw["stream_options"] = opts
	}

	next, err := json.Marshal(raw)
	if err != nil {
		return normalizedRequest{}, fmt.Errorf("marshal request: %w", err)
	}
	trace := buildDebugTrace(rawForTrace, raw, model, stream)
	return normalizedRequest{body: next, model: model, stream: stream, trace: trace}, nil
}

func cloneMap(in map[string]any) map[string]any {
	data, err := json.Marshal(in)
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func sortTools(raw map[string]any) {
	tools, ok := raw["tools"].([]any)
	if !ok || len(tools) < 2 {
		return
	}
	sort.SliceStable(tools, func(i, j int) bool {
		return toolSortKey(tools[i]) < toolSortKey(tools[j])
	})
	raw["tools"] = tools
}

func toolSortKey(tool any) string {
	item, ok := tool.(map[string]any)
	if !ok {
		return ""
	}
	fn, ok := item["function"].(map[string]any)
	if !ok {
		return ""
	}
	name, _ := fn["name"].(string)
	return name
}

func buildDebugTrace(raw, normalized map[string]any, model string, stream bool) debugTrace {
	rawShape := prefixShape(raw)
	normalizedShape := prefixShape(normalized)
	return debugTrace{
		Time:              time.Now(),
		Model:             model,
		Stream:            stream,
		RawPrefixHash:     rawShape.prefixHash,
		NormPrefixHash:    normalizedShape.prefixHash,
		RawSystemHash:     rawShape.systemHash,
		NormSystemHash:    normalizedShape.systemHash,
		RawToolsHash:      rawShape.toolsHash,
		NormToolsHash:     normalizedShape.toolsHash,
		ToolsChanged:      rawShape.toolsHash != normalizedShape.toolsHash,
		SystemChanged:     rawShape.systemHash != normalizedShape.systemHash,
		RawToolsOrder:     toolsOrder(raw),
		NormToolsOrder:    toolsOrder(normalized),
		RawPreview:        previewJSON(raw),
		NormalizedPreview: previewJSON(normalized),
	}
}

type requestShape struct {
	systemHash string
	toolsHash  string
	prefixHash string
}

func prefixShape(raw map[string]any) requestShape {
	system := systemPrompt(raw)
	tools := raw["tools"]
	toolsJSON, _ := json.Marshal(tools)
	return requestShape{
		systemHash: shortHash(system),
		toolsHash:  shortHash(string(toolsJSON)),
		prefixHash: shortHash(map[string]any{
			"system": system,
			"tools":  string(toolsJSON),
		}),
	}
}

func systemPrompt(raw map[string]any) string {
	messages, ok := raw["messages"].([]any)
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, message := range messages {
		item, ok := message.(map[string]any)
		if !ok || item["role"] != "system" {
			continue
		}
		if content, ok := item["content"].(string); ok {
			b.WriteString(content)
		}
	}
	return b.String()
}

func toolsOrder(raw map[string]any) []string {
	tools, ok := raw["tools"].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(tools))
	for _, tool := range tools {
		out = append(out, toolSortKey(tool))
	}
	return out
}

func shortHash(value any) string {
	data, _ := json.Marshal(value)
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum[:8])
}

func previewJSON(raw map[string]any) json.RawMessage {
	clean := map[string]any{}
	for _, key := range []string{"model", "stream", "stream_options", "messages", "tools"} {
		if value, ok := raw[key]; ok {
			clean[key] = limitValue(value)
		}
	}
	data, _ := json.Marshal(clean)
	return data
}

func limitValue(value any) any {
	switch v := value.(type) {
	case string:
		if len(v) > 2000 {
			return v[:2000] + "...[truncated]"
		}
		return v
	case []any:
		limit := len(v)
		if limit > 8 {
			limit = 8
		}
		out := make([]any, 0, limit)
		for i := 0; i < limit; i++ {
			out = append(out, limitValue(v[i]))
		}
		if len(v) > limit {
			out = append(out, fmt.Sprintf("...[truncated %d items]", len(v)-limit))
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, item := range v {
			out[key] = limitValue(item)
		}
		return out
	default:
		return value
	}
}

func copyHeaders(dst, src http.Header) {
	for key, values := range src {
		if strings.EqualFold(key, "Content-Length") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func isStreamResponse(h http.Header) bool {
	return strings.Contains(strings.ToLower(h.Get("Content-Type")), "text/event-stream")
}

func copyStreamAndCaptureUsage(dst io.Writer, src io.Reader, model string, status int) (requestMetric, error) {
	metric := requestMetric{
		Time:   time.Now(),
		Model:  model,
		Stream: true,
		Status: status,
	}
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if _, err := fmt.Fprintln(dst, line); err != nil {
			return metric, err
		}
		if strings.HasPrefix(line, "data:") {
			updateMetricFromSSEData(&metric, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	return metric, scanner.Err()
}

func updateMetricFromSSEData(metric *requestMetric, data string) {
	if data == "" || data == "[DONE]" {
		return
	}
	var chunk streamChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil || chunk.Usage == nil {
		return
	}
	applyUsage(metric, *chunk.Usage)
}

func metricFromResponse(model string, stream bool, status int, body []byte) requestMetric {
	var parsed bufferedResponse
	if err := json.Unmarshal(body, &parsed); err != nil || parsed.Usage.TotalTokens == 0 {
		return requestMetric{
			Time:   time.Now(),
			Model:  model,
			Stream: stream,
			Status: status,
		}
	}
	metric := requestMetric{
		Time:   time.Now(),
		Model:  model,
		Stream: stream,
		Status: status,
	}
	applyUsage(&metric, parsed.Usage)
	return metric
}

func applyUsage(metric *requestMetric, u usage) {
	hitRate := 0.0
	if u.PromptTokens > 0 {
		hitRate = float64(u.PromptCacheHitTokens) / float64(u.PromptTokens)
	}
	metric.PromptTokens = u.PromptTokens
	metric.CompletionTokens = u.CompletionTokens
	metric.TotalTokens = u.TotalTokens
	metric.PromptCacheHitTokens = u.PromptCacheHitTokens
	metric.PromptCacheMissTokens = u.PromptCacheMissTokens
	metric.HitRate = hitRate
	metric.EstimatedCostCNY = estimateCostCNY(u)
	metric.EstimatedSavedCNY = estimateSavedCNY(u)
}

type pricing struct {
	CacheHit float64
	Input    float64
	Output   float64
}

func pricingForModel() pricing {
	return pricing{
		CacheHit: 0.02,
		Input:    1,
		Output:   2,
	}
}

func estimateCostCNY(u usage) float64 {
	rate := pricingForModel()
	return (float64(u.PromptCacheHitTokens)*rate.CacheHit +
		float64(u.PromptCacheMissTokens)*rate.Input +
		float64(u.CompletionTokens)*rate.Output) / 1_000_000
}

func estimateSavedCNY(u usage) float64 {
	if u.PromptCacheHitTokens <= 0 {
		return 0
	}
	rate := pricingForModel()
	return float64(u.PromptCacheHitTokens) * (rate.Input - rate.CacheHit) / 1_000_000
}

func logUsage(m requestMetric) {
	if m.TotalTokens == 0 {
		log.Printf("model=%s stream=%v status=%d", m.Model, m.Stream, m.Status)
		return
	}
	log.Printf(
		"model=%s stream=%v status=%d total=%d prompt=%d cached=%d new=%d completion=%d",
		m.Model,
		m.Stream,
		m.Status,
		m.TotalTokens,
		m.PromptTokens,
		m.PromptCacheHitTokens,
		m.PromptCacheMissTokens,
		m.CompletionTokens,
	)
}

func attachTrace(metric *requestMetric, trace debugTrace) {
	metric.RawPrefixHash = trace.RawPrefixHash
	metric.NormalizedPrefixHash = trace.NormPrefixHash
	metric.ToolsChanged = trace.ToolsChanged
	metric.SystemChanged = trace.SystemChanged
}

func newMetricsStore(traceDir string) *metricsStore {
	store := &metricsStore{traceDir: traceDir}
	if err := store.load(); err != nil {
		log.Printf("trace load skipped: %v", err)
	}
	return store
}

func (s *metricsStore) add(metric requestMetric, trace debugTrace) {
	s.mu.Lock()
	s.nextID++
	metric.ID = s.nextID
	trace.ID = metric.ID
	record := persistedRequest{Metric: metric, Trace: trace}
	s.requests = append(s.requests, metric)
	if len(s.requests) > 200 {
		s.requests = s.requests[len(s.requests)-200:]
	}
	s.debug = append(s.debug, trace)
	if len(s.debug) > 50 {
		s.debug = s.debug[len(s.debug)-50:]
	}
	s.mu.Unlock()

	if err := s.persist(record); err != nil {
		log.Printf("trace persist failed: %v", err)
	}
}

func (s *metricsStore) persist(record persistedRequest) error {
	if strings.TrimSpace(s.traceDir) == "" {
		return nil
	}
	if err := os.MkdirAll(s.traceDir, 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(filepath.Join(s.traceDir, "requests.jsonl"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	if _, err := file.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

func (s *metricsStore) load() error {
	if strings.TrimSpace(s.traceDir) == "" {
		return nil
	}
	data, err := os.ReadFile(filepath.Join(s.traceDir, "requests.jsonl"))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) > 200 {
		lines = lines[len(lines)-200:]
	}
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var record persistedRequest
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			continue
		}
		s.requests = append(s.requests, record.Metric)
		s.debug = append(s.debug, record.Trace)
		if record.Metric.ID > s.nextID {
			s.nextID = record.Metric.ID
		}
	}
	if len(s.debug) > 50 {
		s.debug = s.debug[len(s.debug)-50:]
	}
	return nil
}

func (s *metricsStore) snapshot() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	requests := append([]requestMetric(nil), s.requests...)
	totalPrompt := 0
	totalCached := 0
	totalNew := 0
	totalCost := 0.0
	totalSaved := 0.0
	for _, item := range requests {
		totalPrompt += item.PromptTokens
		totalCached += item.PromptCacheHitTokens
		totalNew += item.PromptCacheMissTokens
		totalCost += item.EstimatedCostCNY
		totalSaved += item.EstimatedSavedCNY
	}
	hitRate := 0.0
	if totalPrompt > 0 {
		hitRate = float64(totalCached) / float64(totalPrompt)
	}
	return map[string]any{
		"summary": map[string]any{
			"requests":     len(requests),
			"promptTokens": totalPrompt,
			"cachedTokens": totalCached,
			"newTokens":    totalNew,
			"hitRate":      hitRate,
			"costCNY":      totalCost,
			"savedCNY":     totalSaved,
		},
		"requests": requests,
	}
}

func (s *metricsStore) debugList() []debugTrace {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]debugTrace, len(s.debug))
	copy(out, s.debug)
	for i := range out {
		out[i].RawPreview = nil
		out[i].NormalizedPreview = nil
	}
	return out
}

func (s *metricsStore) debugByID(id int) (debugTrace, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, item := range s.debug {
		if item.ID == id {
			return item, true
		}
	}
	return debugTrace{}, false
}

func handleMetrics(metrics *metricsStore, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, metrics.snapshot())
}

func handleDebugRequests(metrics *metricsStore, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"requests": metrics.debugList()})
}

func handleDebugRequest(metrics *metricsStore, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	rawID := strings.TrimPrefix(r.URL.Path, "/debug/requests/")
	var id int
	if _, err := fmt.Sscanf(rawID, "%d", &id); err != nil || id <= 0 {
		http.Error(w, "invalid request id", http.StatusBadRequest)
		return
	}
	trace, ok := metrics.debugByID(id)
	if !ok {
		http.Error(w, "debug request not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, trace)
}

func handleDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(dashboardHTML))
}

func handleDebugPage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	http.Redirect(w, r, "/dashboard#debug", http.StatusFound)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

const dashboardHTML = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepSeek Cost Proxy</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #1d232b; }
    header { padding: 20px 28px; background: #111827; color: white; }
    main { padding: 24px 28px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 18px; }
    .tab { border: 1px solid #cbd5e1; background: white; padding: 10px 14px; border-radius: 6px; cursor: pointer; }
    .tab.active { background: #111827; color: white; border-color: #111827; }
    .view { display: none; }
    .view.active { display: block; }
    .grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
    .card { background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; }
    .label { color: #667085; font-size: 13px; }
    .value { font-size: 28px; font-weight: 700; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; margin-top: 18px; background: white; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }
    th { background: #f1f5f9; color: #475569; }
    .bar { height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden; width: 120px; }
    .bar > span { display: block; height: 100%; background: #16a34a; }
    .debug { grid-template-columns: 320px 1fr; gap: 16px; }
    .view.debug.active { display: grid; }
    .debug-list button { display: block; width: 100%; margin: 0 0 8px; padding: 10px; border: 1px solid #d0d5dd; background: white; text-align: left; border-radius: 6px; cursor: pointer; }
    .panel { background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; }
    .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
    .meta div { padding: 8px; background: #f8fafc; border-radius: 6px; overflow-wrap: anywhere; }
    pre { white-space: pre-wrap; overflow: auto; background: #0f172a; color: #e5e7eb; padding: 12px; border-radius: 6px; max-height: 420px; }
    @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 900px) { .debug { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>DeepSeek Cost Proxy</h1>
  </header>
  <main>
    <nav class="tabs">
      <button class="tab active" id="tabDashboard" onclick="showView('dashboard')">Dashboard</button>
      <button class="tab" id="tabDebug" onclick="showView('debug')">Prompt Debug</button>
    </nav>
    <section class="view active" id="dashboardView">
      <section class="grid">
        <div class="card"><div class="label">Requests</div><div class="value" id="requests">0</div></div>
        <div class="card"><div class="label">Prompt Tokens</div><div class="value" id="prompt">0</div></div>
        <div class="card"><div class="label">Cached / New</div><div class="value" id="cached">0 / 0</div></div>
        <div class="card"><div class="label">Hit Rate</div><div class="value" id="hitRate">0%</div></div>
        <div class="card"><div class="label">Estimated Cost</div><div class="value" id="cost">CNY 0</div></div>
        <div class="card"><div class="label">Estimated Saved</div><div class="value" id="saved">CNY 0</div></div>
      </section>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Time</th><th>Model</th><th>Status</th><th>Prompt</th>
            <th>Cached</th><th>New</th><th>Hit</th><th>Stream</th>
            <th>Raw Prefix</th><th>Normalized Prefix</th><th>Tools</th><th>Saved</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
    <section class="view debug" id="debugView">
      <aside class="panel">
        <h2>Requests</h2>
        <div class="debug-list" id="debugList"></div>
      </aside>
      <section class="panel">
        <h2>Raw vs Normalized</h2>
        <div class="meta" id="debugMeta"></div>
        <h3>Raw Hermes Request Preview</h3>
        <pre id="rawPreview">{}</pre>
        <h3>Normalized DeepSeek Request Preview</h3>
        <pre id="normalizedPreview">{}</pre>
      </section>
    </section>
  </main>
  <script>
    const fmt = new Intl.NumberFormat();
    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }
    function pct(v) { return ((v || 0) * 100).toFixed(1) + '%'; }
    function showView(name) {
      const debug = name === 'debug';
      document.querySelector('#dashboardView').classList.toggle('active', !debug);
      document.querySelector('#debugView').classList.toggle('active', debug);
      document.querySelector('#tabDashboard').classList.toggle('active', !debug);
      document.querySelector('#tabDebug').classList.toggle('active', debug);
      location.hash = debug ? 'debug' : 'dashboard';
      if (debug) loadDebugList();
    }
    async function refresh() {
      const res = await fetch('/metrics');
      const data = await res.json();
      const s = data.summary;
      document.querySelector('#requests').textContent = fmt.format(s.requests);
      document.querySelector('#prompt').textContent = fmt.format(s.promptTokens);
      document.querySelector('#cached').textContent = fmt.format(s.cachedTokens) + ' / ' + fmt.format(s.newTokens);
      document.querySelector('#hitRate').textContent = pct(s.hitRate);
      document.querySelector('#cost').textContent = 'CNY ' + (s.costCNY || 0).toFixed(6);
      document.querySelector('#saved').textContent = 'CNY ' + (s.savedCNY || 0).toFixed(6);
      const rows = [...data.requests].reverse().map(item => {
        return '<tr>' +
          '<td>' + item.id + '</td>' +
          '<td>' + new Date(item.time).toLocaleTimeString() + '</td>' +
          '<td>' + esc(item.model) + '</td>' +
          '<td>' + item.status + '</td>' +
          '<td>' + fmt.format(item.promptTokens || 0) + '</td>' +
          '<td>' + fmt.format(item.promptCacheHitTokens || 0) + '</td>' +
          '<td>' + fmt.format(item.promptCacheMissTokens || 0) + '</td>' +
          '<td><div class="bar"><span style="width:' + ((item.hitRate || 0) * 100) + '%"></span></div> ' + pct(item.hitRate) + '</td>' +
          '<td>' + (item.stream ? 'yes' : 'no') + '</td>' +
          '<td>' + esc(item.rawPrefixHash || '') + '</td>' +
          '<td>' + esc(item.normalizedPrefixHash || '') + '</td>' +
          '<td>' + (item.toolsChanged ? 'sorted' : 'same') + '</td>' +
          '<td>CNY ' + (item.estimatedSavedCNY || 0).toFixed(6) + '</td>' +
        '</tr>';
      }).join('');
      document.querySelector('#rows').innerHTML = rows;
    }
    async function loadDebugList() {
      const res = await fetch('/debug/requests');
      const data = await res.json();
      const items = [...data.requests].reverse();
      document.querySelector('#debugList').innerHTML = items.map(item =>
        '<button onclick="loadDebugRequest(' + item.id + ')">#' + item.id + ' ' + esc(item.model) + '<br>' +
        '<small>' + esc(item.rawPrefixHash) + ' -> ' + esc(item.normalizedPrefixHash) + '</small></button>'
      ).join('');
      if (items[0]) loadDebugRequest(items[0].id);
    }
    async function loadDebugRequest(id) {
      const res = await fetch('/debug/requests/' + id);
      const item = await res.json();
      document.querySelector('#debugMeta').innerHTML =
        '<div>Raw Prefix<br><b>' + esc(item.rawPrefixHash) + '</b></div>' +
        '<div>Normalized Prefix<br><b>' + esc(item.normalizedPrefixHash) + '</b></div>' +
        '<div>Tools Changed<br><b>' + item.toolsChanged + '</b></div>' +
        '<div>System Changed<br><b>' + item.systemChanged + '</b></div>' +
        '<div>Raw Tools<br><b>' + esc((item.rawToolsOrder || []).join(', ')) + '</b></div>' +
        '<div>Normalized Tools<br><b>' + esc((item.normalizedToolsOrder || []).join(', ')) + '</b></div>';
      document.querySelector('#rawPreview').textContent = JSON.stringify(item.rawPreview, null, 2);
      document.querySelector('#normalizedPreview').textContent = JSON.stringify(item.normalizedPreview, null, 2);
    }
    if (location.hash === '#debug') showView('debug');
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>`
