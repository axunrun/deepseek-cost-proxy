param(
  [string]$ProxyUrl = "http://127.0.0.1:18188",
  [string]$ProxyKey = "local-proxy-key",
  [string]$Model = "deepseek-v4-flash"
)

$ErrorActionPreference = "Stop"

function New-TestBody {
  param([string]$UserText)

  $stablePolicy = @"
You are a coding agent connected through DeepSeek Cost Proxy.
Keep this system prompt byte-stable for cache testing.
Prefer minimal diffs, inspect files before editing, and preserve tool schemas.
This repeated paragraph intentionally creates a long stable prefix.
"@

  $longSystem = ($stablePolicy + "`n") * 120

  return @{
    model = $Model
    stream = $false
    messages = @(
      @{ role = "system"; content = $longSystem },
      @{ role = "user"; content = $UserText }
    )
    tools = @(
      @{
        type = "function"
        function = @{
          name = "read_file"
          description = "Read a file"
          parameters = @{
            type = "object"
            properties = @{ path = @{ type = "string" } }
            required = @("path")
          }
        }
      },
      @{
        type = "function"
        function = @{
          name = "grep"
          description = "Search files"
          parameters = @{
            type = "object"
            properties = @{ pattern = @{ type = "string" } }
            required = @("pattern")
          }
        }
      }
    )
  } | ConvertTo-Json -Depth 20
}

function Invoke-ProxyRequest {
  param([string]$Body)

  $headers = @{
    Authorization = "Bearer $ProxyKey"
    "Content-Type" = "application/json"
  }

  return Invoke-RestMethod `
    -Uri "$ProxyUrl/v1/chat/completions" `
    -Method Post `
    -Headers $headers `
    -Body $Body
}

function To-Row {
  param([int]$Index, $Response)

  $usage = $Response.usage
  $hitRate = 0
  if ($usage.prompt_tokens -gt 0) {
    $hitRate = [math]::Round(($usage.prompt_cache_hit_tokens / $usage.prompt_tokens) * 100, 2)
  }

  return [pscustomobject]@{
    request = $Index
    prompt = $usage.prompt_tokens
    cached = $usage.prompt_cache_hit_tokens
    new = $usage.prompt_cache_miss_tokens
    completion = $usage.completion_tokens
    hitRate = "$hitRate%"
  }
}

Write-Host "Running cache test against $ProxyUrl with model $Model"

$first = Invoke-ProxyRequest -Body (New-TestBody "First request: answer with one short sentence.")
Start-Sleep -Seconds 2
$second = Invoke-ProxyRequest -Body (New-TestBody "Second request: answer with one short sentence.")

@(
  To-Row -Index 1 -Response $first
  To-Row -Index 2 -Response $second
) | Format-Table -AutoSize

Write-Host ""
Write-Host "Open dashboard: $ProxyUrl/dashboard"
Write-Host "Open debug list: $ProxyUrl/debug/requests"
