
```bash
(echo -n '{ "model": "qwen3.6:35b-a3b-q4_K_M", "messages": [{ "role": "user", "content": [{ "type": "text", "text": "What is in this image?" }, { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,'; base64 -w 0 ./image.jpg; echo '"}}]}]}') | curl -H "Content-Type: application/json"  -d @- http://localhost:11434/v1/chat/completions

```