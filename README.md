# medllm

A small command-line tool for sharing medical vision-language models on a multi-GPU Linux machine.
It starts [vLLM](https://github.com/vllm-project/vllm) in Docker on whichever GPUs are free, stops it
when idle, and gives users a terminal chat, a one-shot `ask`, and a local OpenAI-compatible API.
Each user logs in with their own Hugging Face account, so gated-model licenses (such as MedGemma) are checked per person.

Single Python file, standard library only.

> **Research use only.** Model outputs are not clinical advice. Use de-identified data unless your approvals say otherwise.

## Models

| Alias | Hugging Face repo | GPUs | License step |
|---|---|---|---|
| `medgemma-4b` | `google/medgemma-4b-it` | 1 | accept on Hugging Face |
| `lingshu-7b` | `lingshu-medical-mllm/Lingshu-7B` | 1 | none |
| `medgemma-27b` | `google/medgemma-27b-it` | 4 | accept on Hugging Face |
| `qwen3-vl-32b` | `Qwen/Qwen3-VL-32B-Instruct` | 4 | none |

Edit the `MODELS` table at the top of `medllm` to add or change models.

Throughput measured with `bench.py` on 8x RTX PRO 6000 Blackwell (96 GB), vLLM, 256-token answers:

| | 1 user | 32 concurrent (total) |
|---|---|---|
| medgemma-4b (1 GPU) | 153 tok/s | 3,941 tok/s |
| lingshu-7b (1 GPU) | 89 tok/s | 1,966 tok/s |
| medgemma-27b (4 GPU) | 71 tok/s | 1,531 tok/s |
| qwen3-vl-32b (4 GPU) | 57 tok/s | 1,453 tok/s |

These are speed figures, not quality figures.

## Server setup (once)

Requirements: Linux, NVIDIA GPUs and driver, Docker with the NVIDIA container toolkit, Python 3 (tested on 3.12).
The layout assumes GPUs 0-3 and 4-7 are two 4-GPU groups (edit `GROUPS` for other machines).

```bash
export MEDLLM_ROOT=/mnt/data/llm        # default; any large shared directory
mkdir -p $MEDLLM_ROOT/hf-cache $MEDLLM_ROOT/run
chmod 2775 $MEDLLM_ROOT $MEDLLM_ROOT/run   # group-writable so users can start models and share state

docker pull vllm/vllm-openai:latest      # use a recent image; Blackwell GPUs need CUDA 12.8+

# download weights into the shared cache (gated repos need a token whose account accepted the license)
HF_HOME=$MEDLLM_ROOT/hf-cache hf download lingshu-medical-mllm/Lingshu-7B
HF_HOME=$MEDLLM_ROOT/hf-cache HF_TOKEN=... hf download google/medgemma-27b-it
```

Containers run with `HF_HUB_OFFLINE=1`, so the token used for downloading is never given to them.

## Use

```bash
ln -s "$PWD/medllm" ~/.local/bin/medllm
medllm login                  # verify your own HF account (token is not stored)
medllm status                 # GPUs and running models
medllm serve medgemma-27b     # start on free GPUs; auto-stops after 60 idle minutes
medllm chat medgemma-27b      # /image path, /system text, /clear, /exit
medllm ask -m medgemma-27b -i scan.png "Describe the findings."
medllm ask -m medgemma-4b "Summarize:" - < note.txt     # '-' reads stdin
eval "$(medllm env medgemma-27b)"                      # OPENAI_BASE_URL / OPENAI_API_KEY for scripts
medllm stop medgemma-27b
```

`serve` and `stop` need Docker access. `chat`, `ask`, `status`, `models` and `env` only talk to the local API and shared state files.

## How it works

- **GPU choice.** A GPU is free if it uses under 2 GB and no other instance has reserved it. A 4-GPU model needs a whole group; 1-GPU models fill the busier group first so the other stays free. Selection runs under an `flock`.
- **Idle shutdown.** A detached watcher polls vLLM's `/metrics` and removes the container after `--idle` minutes without requests.
- **Access.** Servers bind to `127.0.0.1` and require a bearer key (passed through `VLLM_API_KEY`, so it is not visible in `ps`). `medllm login` checks the user's own HF license access. The key file is readable by the shared Unix group, so this is not strong isolation between users on the same machine.
- **Logging.** `run/usage.log` records username, model, command and time. Prompts and answers are not recorded.
- **Docker access is root-equivalent.** Only add trusted users to the `docker` group.

## Benchmark

```bash
python3 bench.py google/medgemma-4b-it --gpus 0 --port 18000
```

Starts a container, measures startup time, VRAM, time to first token, single-stream and concurrent throughput, saves four sample medical answers to `$MEDLLM_ROOT/bench/`, and removes the container.

## License

MIT. Model weights have their own licenses; MedGemma is subject to Google's Health AI Developer Foundations terms.
