```bash
hf download hf://Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q4_K_M.gguf

hf download hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf
hf download hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf


hf download hf://Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp/Qwen3-Reranker-4B-Q4_K_M.gguf

llama-server \
    --host 127.0.0.1 \
    --port 8081 \
    --metrics \
    --models-max 1 \
    --models-preset models.ini

    # failed to load model Qwen3-VL-8B-Instruct-GGUF
```

```bash
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-4B-Q4_K_M",
    "input": ["The Labour Code requires 30 calendar days written notice."]
  }'
```

```bash
curl http://localhost:8081/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-4B-Q4_K_M",
    "query": "employment termination notice period",
    "documents": [
      "The Labour Code requires 30 calendar days written notice.",
      "Corporate tax rates for small enterprises."
    ]
  }'
  ```

```bash
curl http://localhost:8081/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen3VL-8B-Instruct-Q8_0",
"messages": [
    {"role": "user", "content": "Hello!"}
],
"max_tokens": 128
}'
```