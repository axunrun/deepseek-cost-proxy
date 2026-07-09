FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/deepseek-cost-proxy .

FROM alpine:3.20
COPY --from=build /out/deepseek-cost-proxy /usr/local/bin/deepseek-cost-proxy
ENV PROXY_ADDR=18188 \
    DEFAULT_MODEL=deepseek-v4-flash \
    MINIMAX_CHAT_URL=https://api.minimax.io/v1/chat/completions \
    TRACE_DIR=/data/traces
EXPOSE 18188
ENTRYPOINT ["deepseek-cost-proxy"]
