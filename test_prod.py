"""Production-like test suite for qwen3-tts-server."""
import asyncio
import time
import json
import base64
import struct
import aiohttp

BASE = "http://localhost:8880"

results = []

def log(test, status, detail="", duration=0):
    icon = "PASS" if status else "FAIL"
    results.append((test, status))
    dur = f" ({duration:.2f}s)" if duration else ""
    print(f"  [{icon}] {test}{dur} {detail}")


async def test_health(session):
    t = time.time()
    async with session.get(f"{BASE}/health") as r:
        data = await r.json()
        ok = r.status == 200 and data["status"] == "healthy" and data["backend"]["ready"]
        log("Health check", ok, f"device={data['device']['type']}", time.time()-t)


async def test_models(session):
    t = time.time()
    async with session.get(f"{BASE}/v1/models") as r:
        data = await r.json()
        ok = r.status == 200 and len(data.get("data", [])) > 0
        log("List models", ok, f"count={len(data.get('data',[]))}", time.time()-t)


async def test_voices(session):
    t = time.time()
    async with session.get(f"{BASE}/v1/voices") as r:
        data = await r.json()
        voices = data.get("voices", [])
        ok = r.status == 200 and len(voices) > 0
        log("List voices", ok, f"count={len(voices)}", time.time()-t)


async def test_speech_all_voices(session):
    """Test each OpenAI voice."""
    voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
    for voice in voices:
        t = time.time()
        payload = {"model": "qwen3-tts", "input": "Testing voice.", "voice": voice}
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            body = await r.read()
            ok = r.status == 200 and len(body) > 1000
            log(f"Speech voice={voice}", ok, f"size={len(body)}B", time.time()-t)


async def test_speech_formats(session):
    """Test each audio format."""
    formats = ["mp3", "wav", "flac", "opus", "aac", "pcm"]
    for fmt in formats:
        t = time.time()
        payload = {"model": "qwen3-tts", "input": "Format test.", "voice": "alloy", "response_format": fmt}
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            body = await r.read()
            ok = r.status == 200 and len(body) > 100
            detail = f"size={len(body)}B"
            if r.status != 200:
                detail = (await r.json() if r.content_type == 'application/json' else await r.text())[:200]
            log(f"Format {fmt}", ok, detail, time.time()-t)


async def test_speech_speeds(session):
    """Test speed parameter."""
    for speed in [0.5, 1.0, 1.5, 2.0]:
        t = time.time()
        payload = {"model": "qwen3-tts", "input": "Speed test.", "voice": "alloy", "speed": speed}
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            body = await r.read()
            ok = r.status == 200 and len(body) > 100
            log(f"Speed {speed}x", ok, f"size={len(body)}B", time.time()-t)


async def test_speech_instruct(session):
    """Test instruct/style parameter."""
    t = time.time()
    payload = {
        "model": "qwen3-tts",
        "input": "This is exciting news!",
        "voice": "alloy",
        "instruct": "Speak with enthusiasm and energy"
    }
    async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
        body = await r.read()
        ok = r.status == 200 and len(body) > 100
        detail = f"size={len(body)}B" if ok else str(await r.text())[:200]
        log("Instruct/style", ok, detail, time.time()-t)


async def test_long_text(session):
    """Test with longer text."""
    t = time.time()
    text = "This is a longer piece of text to test how the system handles more substantial input. " * 5
    payload = {"model": "qwen3-tts", "input": text.strip(), "voice": "alloy"}
    async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
        body = await r.read()
        ok = r.status == 200 and len(body) > 5000
        log("Long text", ok, f"size={len(body)}B, input={len(text)}chars", time.time()-t)


async def test_empty_input(session):
    """Test empty/invalid input."""
    t = time.time()
    payload = {"model": "qwen3-tts", "input": "", "voice": "alloy"}
    async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
        ok = r.status in [400, 422]  # should reject empty input
        log("Empty input rejected", ok, f"status={r.status}", time.time()-t)


async def test_invalid_voice(session):
    """Test invalid voice name."""
    t = time.time()
    payload = {"model": "qwen3-tts", "input": "Test.", "voice": "nonexistent_voice_xyz"}
    async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
        ok = r.status in [400, 422]
        log("Invalid voice rejected (400)", ok, f"status={r.status}", time.time()-t)


async def test_concurrent_requests(session):
    """Test concurrent requests - do they run simultaneously or queue?"""
    text = "Concurrent test sentence number one two three."
    payloads = [
        {"model": "qwen3-tts", "input": text, "voice": v}
        for v in ["alloy", "echo", "fable"]
    ]

    t_start = time.time()

    async def single_req(payload, idx):
        t = time.time()
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            body = await r.read()
            elapsed = time.time() - t
            return idx, r.status, len(body), elapsed

    tasks = [single_req(p, i) for i, p in enumerate(payloads)]
    results_conc = await asyncio.gather(*tasks, return_exceptions=True)
    total = time.time() - t_start

    all_ok = True
    individual_times = []
    for res in results_conc:
        if isinstance(res, Exception):
            print(f"    Exception: {res}")
            all_ok = False
        else:
            idx, status, size, elapsed = res
            individual_times.append(elapsed)
            if status != 200:
                all_ok = False
            print(f"    Request {idx}: HTTP {status}, {size}B, {elapsed:.2f}s")

    avg_individual = sum(individual_times) / len(individual_times) if individual_times else 0

    # If total time is close to max individual time, they ran concurrently
    # If total time is close to sum of individual times, they queued
    if individual_times:
        max_t = max(individual_times)
        sum_t = sum(individual_times)
        if total < sum_t * 0.75:
            mode = "CONCURRENT (parallel)"
        else:
            mode = "SEQUENTIAL (queued)"
        print(f"    Total wall time: {total:.2f}s, sum of individual: {sum_t:.2f}s -> {mode}")

    log("Concurrent requests (3x)", all_ok, f"wall={total:.2f}s", total)


async def test_concurrent_heavy(session):
    """Test 5 concurrent requests."""
    payloads = [
        {"model": "qwen3-tts", "input": f"Heavy concurrent test number {i}.", "voice": "alloy"}
        for i in range(5)
    ]

    t_start = time.time()

    async def single_req(payload, idx):
        t = time.time()
        try:
            async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
                body = await r.read()
                return idx, r.status, len(body), time.time() - t
        except Exception as e:
            return idx, 0, 0, time.time() - t

    tasks = [single_req(p, i) for i, p in enumerate(payloads)]
    results_conc = await asyncio.gather(*tasks)
    total = time.time() - t_start

    all_ok = True
    for idx, status, size, elapsed in results_conc:
        if status != 200:
            all_ok = False
        print(f"    Request {idx}: HTTP {status}, {size}B, {elapsed:.2f}s")

    sum_t = sum(r[3] for r in results_conc)
    mode = "CONCURRENT" if total < sum_t * 0.75 else "SEQUENTIAL"
    print(f"    Total wall: {total:.2f}s, sum individual: {sum_t:.2f}s -> {mode}")

    log("Concurrent requests (5x)", all_ok, f"wall={total:.2f}s", total)


async def test_rapid_sequential(session):
    """Rapid sequential requests - no delay between them."""
    t_start = time.time()
    all_ok = True
    for i in range(5):
        payload = {"model": "qwen3-tts", "input": f"Rapid test {i}.", "voice": "alloy"}
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            if r.status != 200:
                all_ok = False
            await r.read()
    total = time.time() - t_start
    log("Rapid sequential (5x)", all_ok, f"total={total:.2f}s", total)


async def test_special_characters(session):
    """Test text with special chars, numbers, URLs, etc."""
    texts = [
        ("Numbers & symbols", "The price is $49.99 for 3 items at 10% off."),
        ("URL", "Visit https://example.com for more info."),
        ("Email", "Send it to user@example.com please."),
        ("Punctuation", "Wait... really?! That's amazing! Wow!!!"),
        ("Unicode", "Café résumé naïve"),
    ]
    for name, text in texts:
        t = time.time()
        payload = {"model": "qwen3-tts", "input": text, "voice": "alloy"}
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            body = await r.read()
            ok = r.status == 200 and len(body) > 100
            log(f"Special: {name}", ok, f"size={len(body)}B", time.time()-t)


async def test_direct_qwen_voices(session):
    """Test using Qwen voice names directly (not OpenAI aliases)."""
    qwen_voices = ["Vivian", "Ryan", "Serena", "Sohee", "Eric", "Ono_anna", "Aiden", "Dylan", "Uncle_fu"]
    for voice in qwen_voices:
        t = time.time()
        payload = {"model": "qwen3-tts", "input": "Direct voice test.", "voice": voice}
        async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
            body = await r.read()
            ok = r.status == 200 and len(body) > 100
            detail = f"size={len(body)}B" if ok else f"status={r.status}"
            log(f"Qwen voice={voice}", ok, detail, time.time()-t)


async def main():
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print("\n=== BASIC ENDPOINTS ===")
        await test_health(session)
        await test_models(session)
        await test_voices(session)

        print("\n=== ALL OPENAI VOICES ===")
        await test_speech_all_voices(session)

        print("\n=== ALL QWEN VOICES (direct) ===")
        await test_direct_qwen_voices(session)

        print("\n=== AUDIO FORMATS ===")
        await test_speech_formats(session)

        print("\n=== SPEED CONTROL ===")
        await test_speech_speeds(session)

        print("\n=== INSTRUCT/STYLE ===")
        await test_speech_instruct(session)

        print("\n=== SPECIAL CHARACTERS ===")
        await test_special_characters(session)

        print("\n=== LONG TEXT ===")
        await test_long_text(session)

        print("\n=== ERROR HANDLING ===")
        await test_empty_input(session)
        await test_invalid_voice(session)

        print("\n=== CONCURRENCY (3 parallel) ===")
        await test_concurrent_requests(session)

        print("\n=== CONCURRENCY (5 parallel) ===")
        await test_concurrent_heavy(session)

        print("\n=== RAPID SEQUENTIAL (5x) ===")
        await test_rapid_sequential(session)

    # Summary
    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed, {len(results)} total")
    if failed:
        print("FAILURES:")
        for name, s in results:
            if not s:
                print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())
