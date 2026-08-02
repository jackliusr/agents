import time

first_token_time = None
last_token_time = None

def timing_callback(chunk):
    global first_token_time, last_token_time
    now = time.perf_counter()
    if first_token_time is None:
        first_token_time = now
    last_token_time = now

from transformers import DynamicCache
from turboquant_vllm import CompressedDynamicCache

# The CompressedDynamicCache modifies the DynamicCache internally,
# so we pass the same `cache` instance to both the generator,
# and not `compressed` directly.
cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)


from haystack_integrations.components.generators.transformers import TransformersChatGenerator

generator = TransformersChatGenerator(
    model="Qwen/Qwen3-4B-Thinking-2507",
    task="text-generation",
    generation_kwargs={
        "past_key_values": cache,
        "use_cache": True,
    },
    streaming_callback=timing_callback,
)

from haystack.dataclasses import ChatMessage

start = time.perf_counter()
output = generator.run(messages=[
    ChatMessage.from_user("What is the capital of France?"),
])
total_time = time.perf_counter() - start
reply = output["replies"][0]
print(reply.text)

tokens = reply.meta["usage"]["completion_tokens"]
if first_token_time is not None and last_token_time is not None:
    generation_time = last_token_time - first_token_time
    print(f"TTFT: {first_token_time - start:.3f}s")
    print(f"Tokens: {tokens}")
    print(f"Speed: {tokens / generation_time:.1f} tok/s")
print(f"Total time: {total_time:.3f}s")

print(compressed.vram_bytes())
