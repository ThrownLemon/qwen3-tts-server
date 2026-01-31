import time, requests

text = ("The quick brown fox jumps over the lazy dog near the river bank. " * 100)[:4096]
print(f"Input length: {len(text)} chars")

t = time.time()
r = requests.post("http://localhost:8880/v1/audio/speech", json={"model": "qwen3-tts", "input": text, "voice": "alloy"})
elapsed = time.time() - t

print(f"Status: {r.status_code}")
print(f"Time: {elapsed:.1f}s")
print(f"Audio size: {len(r.content)} bytes ({len(r.content)/1024:.0f} KB)")
if r.status_code != 200:
    print(f"Error: {r.text[:500]}")
