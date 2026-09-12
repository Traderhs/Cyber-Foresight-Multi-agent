$ErrorActionPreference = 'Stop'

$LlamaRoot = 'C:\Users\Lab-AXIS\Desktop\Haseung Ryu\LLMs\llama.cpp-b10919-cuda13.3\runtime'
$LlamaServer = Join-Path $LlamaRoot 'llama-server.exe'
$Model = 'C:\Users\Lab-AXIS\Desktop\Haseung Ryu\LLMs\Qwen3.8-27B-Q6-K-L\Model\Qwen3.8-27B-Q6_K_L.gguf'

if (-not (Test-Path -LiteralPath $LlamaServer)) {
    throw "llama-server.exe not found: $LlamaServer"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Qwen3.8 model not found: $Model"
}

# Qwen3.8 official thinking-mode sampling:
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
# presence_penalty=0.0, repetition_penalty=1.0.
#
# Stage 1 still fans out logically in LangGraph, but llama-server deliberately
# uses one inference slot. Current llama.cpp CUDA reports have shown corruption
# with concurrent long MTP generations; serial server execution preserves the
# independent-input contract without allowing one critic to observe the other.
& $LlamaServer `
    --model $Model `
    --alias 'Qwen3.8-27B-Q6_K_L' `
    --host 127.0.0.1 `
    --port 8080 `
    --no-webui `
    --metrics `
    --n-gpu-layers all `
    --ctx-size 131072 `
    --parallel 1 `
    --batch-size 2048 `
    --ubatch-size 512 `
    --flash-attn on `
    --cache-type-k q8_0 `
    --cache-type-v q8_0 `
    --spec-draft-type-k q8_0 `
    --spec-draft-type-v q8_0 `
    --cache-reuse 0 `
    --temp 1.0 `
    --top-p 0.95 `
    --top-k 20 `
    --min-p 0.0 `
    --presence-penalty 0.0 `
    --repeat-penalty 1.0 `
    --reasoning on `
    --reasoning-effort xhigh `
    --reasoning-budget -1 `
    --reasoning-format deepseek `
    --reasoning-preserve `
    --spec-type draft-mtp `
    --spec-draft-n-max 4 `
    --spec-draft-n-min 0 `
    --spec-draft-p-min 0.05 `
    --fit off `
    --no-context-shift `
    --jinja

