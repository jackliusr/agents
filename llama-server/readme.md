```bash
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-4B-Q4_K_M",
    "input": ["Your text to embed"]
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