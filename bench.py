#!/usr/bin/env python3
"""Benchmark a model served by vLLM in Docker on one 4-GPU group.

usage: bench.py <hf-model-id> [--gpus 0,1,2,3] [--port 8000] [--max-len 32768]
Starts the container, waits for readiness, measures load time, VRAM, TTFT,
single-stream and concurrent throughput, runs a few medical prompts, stops it.
Results -> $MEDLLM_ROOT/bench/<model>.json
"""
import argparse, json, subprocess, sys, time, threading, urllib.request, os

ROOT = os.environ.get("MEDLLM_ROOT", "/mnt/data/llm")
CACHE = f"{ROOT}/hf-cache"
IMAGE = "vllm/vllm-openai:latest"

PROMPTS = [
    "A 58-year-old presents with crushing chest pain radiating to the left arm and diaphoresis. List the top differential diagnoses and the first three workup steps.",
    "Explain the difference between type 1 and type 2 respiratory failure with an example of each.",
    "What are the typical MRI findings of multiple sclerosis, and which sequences are most useful?",
    "Summarize the mechanism of action of metformin and its major contraindications.",
]


def post(port, body, stream=False, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def gen(port, model, prompt, max_tokens=256):
    """Return (ttft, total_time, n_tokens, text) using streaming."""
    body = dict(model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0, stream=True,
                stream_options={"include_usage": True})
    t0 = time.time(); ttft = None; text = ""; ntok = 0
    with post(port, body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            for c in d.get("choices", []):
                delta = c.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    text += delta
    return ttft, time.time() - t0, ntok, text


def vram(gpus):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    want = set(gpus.split(","))
    return {l.split(",")[0].strip(): int(l.split(",")[1]) for l in out.strip().splitlines()
            if l.split(",")[0].strip() in want}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=32768)
    a = ap.parse_args()
    name = "bench-" + a.model.replace("/", "_").lower()
    tp = len(a.gpus.split(","))
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    token = os.path.expanduser("~/.cache/huggingface/token")
    env = ["-e", "HF_TOKEN"] if os.path.exists(token) else []
    if env: os.environ["HF_TOKEN"] = open(token).read().strip()
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", '"device=' + a.gpus + '"', "--ipc=host",
           "-p", f"127.0.0.1:{a.port}:8000", "-v", f"{CACHE}:/root/.cache/huggingface", *env,
           IMAGE, "--model", a.model, "--tensor-parallel-size", str(tp),
           "--max-model-len", str(a.max_len), "--gpu-memory-utilization", "0.90",
           "--served-model-name", a.model]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    res = {"model": a.model, "gpus": a.gpus}
    try:
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2); break
            except Exception:
                st = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                                    capture_output=True, text=True).stdout.strip()
                if st != "true":
                    print(subprocess.run(["docker", "logs", "--tail", "40", name],
                                         capture_output=True, text=True).stderr)
                    sys.exit("container died during startup")
                if time.time() - t0 > 1800: sys.exit("startup timeout")
                time.sleep(5)
        res["startup_s"] = round(time.time() - t0, 1)
        res["vram_mib"] = vram(a.gpus)
        gen(a.port, a.model, "Hello", 8)  # warmup
        ttft, tot, n, _ = gen(a.port, a.model, PROMPTS[0], 256)
        res["single"] = {"ttft_s": round(ttft, 3), "tok_per_s": round(n / (tot - ttft), 1), "tokens": n}
        for conc in (8, 32):
            outs = []; lock = threading.Lock()
            def w(i):
                r = gen(a.port, a.model, PROMPTS[i % len(PROMPTS)], 256)
                with lock: outs.append(r)
            t1 = time.time()
            th = [threading.Thread(target=w, args=(i,)) for i in range(conc)]
            [t.start() for t in th]; [t.join() for t in th]
            wall = time.time() - t1
            res[f"conc{conc}"] = {"agg_tok_per_s": round(sum(o[2] for o in outs) / wall, 1),
                                  "mean_ttft_s": round(sum(o[0] for o in outs) / len(outs), 2)}
        res["samples"] = [{"prompt": p, "answer": gen(a.port, a.model, p, 400)[3]} for p in PROMPTS]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    os.makedirs(f"{ROOT}/bench", exist_ok=True)
    path = f"{ROOT}/bench/{a.model.replace('/', '_')}.json"
    json.dump(res, open(path, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2)); print("->", path)


if __name__ == "__main__":
    main()
