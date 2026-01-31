"""Test the trained Jake voice model. Run after starting the server with start.sh."""
import os
import requests
import time

SERVER = "http://localhost:8880"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

tests = [
    {"name": "jake_short", "voice": "Jake", "text": "Hey, how's it going? Good to see you."},
    {"name": "jake_medium", "voice": "Jake", "text": "I think the most important thing about making movies is the process itself. The creation of something is often more meaningful than the result."},
    {"name": "jake_long", "voice": "Jake", "text": "You know, when I look back at my career, I realize that every role taught me something different about myself. Every character I've played has left a mark on who I am as a person. And I think that's what makes this job so extraordinary."},
    {"name": "jake_emotional", "voice": "Jake", "instruct": "warm and reflective", "text": "Sometimes I think about the people who believed in me when I didn't believe in myself. I owe them everything."},
]

print(f"Testing Jake voice - output to {OUTPUT_DIR}/\n")

# Wait for server
print("Waiting for server...", end=" ", flush=True)
for _ in range(30):
    try:
        r = requests.get(f"{SERVER}/health", timeout=2)
        if r.status_code == 200:
            print("ready!\n")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("server not responding!")
    exit(1)

for t in tests:
    payload = {
        "model": "qwen3-tts",
        "voice": t["voice"],
        "input": t["text"],
        "response_format": "wav",
    }
    if "instruct" in t:
        payload["instruct"] = t["instruct"]

    print(f"  {t['name']}... ", end="", flush=True)
    start = time.time()
    try:
        r = requests.post(f"{SERVER}/v1/audio/speech", json=payload, timeout=120)
        elapsed = time.time() - start
        if r.status_code == 200:
            out_path = os.path.join(OUTPUT_DIR, f"{t['name']}.wav")
            with open(out_path, "wb") as f:
                f.write(r.content)
            size_kb = len(r.content) / 1024
            print(f"OK ({elapsed:.1f}s, {size_kb:.0f}KB) -> {out_path}")
        else:
            print(f"FAILED (HTTP {r.status_code}): {r.text[:200]}")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nDone! Check audio files in: {OUTPUT_DIR}/")
