"""Benchmark against BENCHMARK_RESULTS.md baselines (RTX 3090).
Current GPU: RTX 4090. Using same test prompts and methodology."""
import time
import requests
import statistics

BASE = "http://localhost:8880"

TESTS = [
    ("Short (2 words)", "Hello world"),
    ("Sentence (7 words)", "Kia ora koutou, welcome to today's meeting."),
    ("Medium (20 words)", "The quick brown fox jumps over the lazy dog near the riverbank. This is a test of text-to-speech generation quality."),
    ("Long (36 words)", "Artificial intelligence has revolutionized the way we interact with technology. Text-to-speech technology has advanced significantly in recent years. Modern neural networks can generate remarkably natural-sounding speech. The Qwen3-TTS model represents the latest breakthrough in this field."),
]

# RTX 3090 baselines (Official + Flash Attn 2, warm median)
BASELINES = {
    "Short (2 words)": 1.01,
    "Sentence (7 words)": 3.29,
    "Medium (20 words)": 8.50,
    "Long (36 words)": 21.16,
}

print(f"{'Test Case':<25} {'Cold':>6} {'Warm Med':>9} {'Warm Avg':>9} {'p95':>7} {'3090 Med':>9} {'Speedup':>8}")
print("-" * 85)

for name, text in TESTS:
    times = []
    cold_time = None

    # 1 cold + 5 warm
    for i in range(6):
        t = time.time()
        r = requests.post(f"{BASE}/v1/audio/speech", json={"model": "qwen3-tts", "input": text, "voice": "alloy"})
        elapsed = time.time() - t

        if r.status_code != 200:
            print(f"  ERROR on {name} run {i}: {r.status_code} {r.text[:200]}")
            break

        if i == 0:
            cold_time = elapsed
        else:
            times.append(elapsed)

    if times:
        med = statistics.median(times)
        avg = statistics.mean(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        baseline = BASELINES.get(name, 0)
        speedup = baseline / med if med > 0 else 0

        print(f"{name:<25} {cold_time:>5.2f}s {med:>8.2f}s {avg:>8.2f}s {p95:>6.2f}s {baseline:>8.2f}s {speedup:>7.2f}x")

print()
print("Speedup = RTX 3090 median / RTX 4090 median (>1.0 means 4090 is faster)")
