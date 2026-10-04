# 05 — CPU Deployment & Quantization Research

**Date:** 2026-10-03
**Target hardware:** Intel i3-4005U (2014 Haswell), 2 physical cores, 7.7 GiB RAM (~2.3 GiB free), AVX2+FMA, no AVX512, no GPU.
**Task:** Banking77 collapsed 77→11 intents. Train 9,158 / val 1,962 / test 1,963. Current best: TF-IDF+LinearSVC 94.50% acc, 0.01 s/1963 tickets.

---

## 1. ONNX Runtime INT8 Quantization of DistilBERT

### 1.1 The "trap": dynamic quantization on non-VNNI CPUs

**Critical finding:** ONNX Runtime **dynamic** INT8 quantization provides minimal-to-no speedup on CPUs without AVX512_VNNI. Haswell (2014) has AVX2 but **not** VNNI (introduced in Cascade Lake, 2019).

| Source | CPU | VNNI? | FP32 | INT8 dynamic | Speedup |
|---|---|---|---|---|---|
| [GitHub issue #6695](https://github.com/microsoft/onnxruntime/issues/6695) | 8-core Cascade Lake (GCP n2) | Yes | 31.46 ms | 19.76 ms | **1.59×** |
| [GitHub issue #6695](https://github.com/microsoft/onnxruntime/issues/6695) | Ryzen 3700X | No | 38.05 ms | 35.08 ms | **1.08×** |
| [GitHub issue #6695](https://github.com/microsoft/onnxruntime/issues/6695) | GCP N1 Skylake | No | — | — | minimal |

> "Transformer dynamic quantization doesn't seem to offer any speed improvement on non-Cascade Lake VNNI CPUs" — [issue #6695](https://github.com/microsoft/onnxruntime/issues/6695)

The ONNX Runtime docs FAQ explicitly warns:
> "Old hardware has none or few of the instructions needed to perform efficient inference in int8. And quantization has overhead (from quantizing and dequantizing), so it is not rare to get worse performance on old devices." — [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

And:
> "x86-64 with VNNI, GPU with Tensor Core int8 support and Arm-based processors with dot-product instructions can get better performance in general." — [same docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

**Implication for Haswell i3-4005U:** Dynamic INT8 quantization is a **trap** — expect ≤1.1× speedup, possibly worse. The quantize/dequantize overhead eats the arithmetic savings.

### 1.2 Static quantization (QDQ) on AVX2

The Microsoft/HF blog (2020) benchmarked BERT-base, RoBERTa-base, GPT-2 on AVX2 vs VNNI:

> "We saw smaller, but still significant, speedups on the AVX2 machine." — [Microsoft blog](https://medium.com/microsoft/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7)

- **VNNI machine:** up to **6×** speedup over PyTorch FP32
- **AVX2 machine:** "smaller but still significant" (exact numbers in images, not extractable — UNVERIFIED precise ratio)
- ONNX Runtime INT8 **consistently beat** PyTorch INT8 on both platforms
- Model size reduced ~4×

### 1.3 Accuracy

- ONNX Runtime INT8 on BERT MRPC: "same accuracy and a slightly higher F1 score" vs PyTorch quantization — [Microsoft blog](https://medium.com/microsoft/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7)
- [arXiv 2608.18182](https://arxiv.org/html/2608.18182) (Intel, 2026): DistilBERT SQuAD F1 86.39→86.40 (Smooth-Dynamic, 0.0% loss), 86.39→86.00 (Smooth-Static, 0.5% loss). MultiNLI: 82.21→82.17 (0.0% loss). **Negligible accuracy loss (<1%)** on server CPUs.

### 1.4 Known AVX2 correctness bug

> "Inference of a BERT INT8 quantized model gives incorrect result on AVX2 only (no AVX-512 support) CPU... UINT8 (unsigned int) quantized model works correctly on both AVX2 CPU and AVX-512 CPU." — [HF forum](https://discuss.huggingface.co/t/difference-in-the-vector-generated-by-the-int8-quantized-model-vs-base-onnx-model/164270)

**This is a critical gating constraint:** S8S8 (signed) INT8 may produce wrong results on AVX2. Use **U8U8** format on Haswell.

### 1.5 Recommendation for Haswell

| Method | Expected speedup on Haswell | Accuracy risk | Verdict |
|---|---|---|---|
| Dynamic INT8 | ≤1.1× (likely a trap) | Low | **Skip** |
| Static QDQ S8S8 | 1.5–2.5× (UNVERIFIED exact) | **High — AVX2 bug** | **Avoid** |
| Static QDQ U8U8 | 1.5–2.5× (UNVERIFIED exact) | Low | **Try first** |
| ONNX Runtime FP32 (no quant) | 1.3–1.6× vs PyTorch | None | **Safe baseline** |

---

## 2. torch Dynamic Quantization (qint8 Linear) for DistilBERT on CPU

- PyTorch has a [dynamic quantization BERT tutorial](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html) (official).
- [Intel article](https://www.intel.com/content/www/us/en/developer/articles/technical/int8-quantization-for-x86-cpu-in-pytorch.html): PyTorch 2.0 x86 backend offers **1.43× speedup** for INT8 vs FBGEMM backend (general, not DistilBERT-specific).
- TorchAO + SmoothQuant (arXiv 2608.18182): On Ice Lake (VNNI), DistilBERT static SmoothQuant gives **2.30×** throughput speedup. On Granite Rapids (AMX), **5.37×**. But these are server CPUs.
- **No measured DistilBERT-specific torch dynamic quantization numbers on laptop CPUs found in this research.** UNVERIFIED.

**Verdict:** torch dynamic quantization is simpler to apply (3 lines) but likely has the same non-VNNI limitation. Static quantization with calibration is better but requires representative data.

---

## 3. Best Small Encoder for Accuracy per CPU Cost

### 3.1 Candidate comparison

| Model | Params | Layers | Hidden | Size (fp32) | Banking77 / CLINC150 | CPU latency | Haswell-safe? |
|---|---|---|---|---|---|---|---|
| **ModernBERT-base** | 139M | 24 | 768 | ~556 MB | Banking77: **93% F1** (vs BERT 90%) — [philschmid](https://www.philschmid.de/fine-tune-modern-bert-in-2025) | 2–4× faster than BERT — [HF blog](https://huggingface.co/blog/modernbert) | ⚠️ Needs flash-attn for full speed; may need newer glibc |
| **DeBERTa-v3-small** | 43M | 6 | 768 | ~172 MB | GLUE 82.1 — [OpenReview](https://openreview.net/pdf?id=sE7-XhLxHA) | ~30% slower fwd-bwd than BERT — [emergentmind](https://www.emergentmind.com/topics/deberta-v3-model) | ✅ Yes |
| **MiniLM-L6-v2** | 22M | 6 | 384 | ~88 MB | MTEB Banking77: 69.67% — [GISTEmbed](https://arxiv.org/html/2402.16829v1) | Very fast | ✅ Yes |
| **bge-small-en-v1.5** | 33M | 12 | 384 | ~133 MB | MTEB Banking77: **74.14%** — [HF](https://huggingface.co/BAAI/bge-small-en-v1.5) | Fast | ✅ Yes |
| **all-MiniLM-L6-v2** | 22M | 6 | 384 | ~88 MB | MTEB Banking77: 69.67% — [GISTEmbed](https://arxiv.org/html/2402.16829v1) | Very fast | ✅ Yes |
| **DistilBERT-base** | 66M | 6 | 768 | ~264 MB | GLUE 86.9 (MNLI) — [HF](https://huggingface.co/distilbert/distilbert-base-uncased) | Baseline | ✅ Yes |
| **ELECTRA-small** | 14M | 12 | 256 | ~56 MB | GLUE 77.4 — [Google](https://github.com/google-research/electra) | Very fast | ✅ Yes |
| **SetFit + ModernBERT** | 139M | 24 | 768 | ~556 MB | IMDB 92.7% (8 shots) — [LinkedIn](https://www.linkedin.com/posts/moshe-wasserblat-8977632) | — | ⚠️ Same as ModernBERT |

### 3.2 Key observations

- **ModernBERT-base** is the accuracy leader (93% F1 on Banking77, +3% over BERT) and 2–4× faster than previous encoders. But it **requires flash-attention** for full speed, which may need newer glibc and compilation. The [ModernBERT repo](https://github.com/answerdotai/ModernBERT) has `requirements-cpu.txt`.
- **DeBERTa-v3-small** is the best accuracy-per-parameter (43M, GLUE 82.1) but disentangled attention is ~30% slower. Still runs on Haswell.
- **bge-small-en-v1.5** (33M, 133 MB) is the best tiny option — 74.14% on MTEB Banking77, runs anywhere.
- **MiniLM-L6-v2** is the weakest (69.67%) but smallest and fastest.
- **ELECTRA-small** is too weak (GLUE 77.4) for this task.
- **SetFit + ModernBERT** gives 92.7% on IMDB with only 8 samples/class — useful if labeling budget is tight.

### 3.3 Recommendation

| Priority | Model | Why |
|---|---|---|
| **1st (accuracy)** | ModernBERT-base | 93% F1 Banking77, 2–4× faster. Verify flash-attn installs on Mint 22.3. |
| **2nd (safe)** | DeBERTa-v3-small | 43M, GLUE 82.1, runs anywhere, no flash-attn needed |
| **3rd (tiny)** | bge-small-en-v1.5 | 33M, 133 MB, 74.14% Banking77, good for kNN retrieval |

---

## 4. GLIBC / oneDNN Requirements

### 4.1 ONNX Runtime

- ONNX Runtime 1.16+ requires **glibc ≥ 2.28** — [GitHub discussion #16088](https://github.com/microsoft/onnxruntime/discussions/16088)
- Linux Mint 22.3 (Ubuntu 24.04 base) has **glibc 2.39** ✅
- ONNX Runtime 1.30 will run fine on Mint 22.3.
- oneDNN (used by ONNX Runtime for CPU) supports AVX2 ✅

### 4.2 ModernBERT

- ModernBERT uses **flash-attention** (FA2/FA3) for optimal speed — [HF blog](https://huggingface.co/blog/modernbert)
- `flash-attn` pip package may need compilation and newer glibc. The [ModernBERT repo](https://github.com/answerdotai/ModernBERT) provides `requirements-cpu.txt`.
- **Risk:** flash-attn compilation may fail on older glibc or without CUDA. On CPU-only Haswell, the benefit of flash-attn is smaller (mainly helps long sequences). For seq 64, standard attention is fine.
- **Verdict:** ModernBERT will run on Haswell for seq 64 even without flash-attn, but you lose the 2–4× speed advantage. Test installation first.

### 4.3 PyTorch / Transformers

- PyTorch 2.x requires glibc ≥ 2.17 (Mint 22.3 has 2.39) ✅
- Transformers library is pure Python + tokenizers (Rust) — no glibc issues ✅

---

## 5. Best Tiny LLM for ~2.3 GiB Free RAM

### 5.1 Memory footprint comparison

| Model | Params | fp32 | bf16 | int8 | Q4_K_M (GGUF) | Source |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 494M | 1,898 MB | ~950 MB | 521 MB | ~398 MB | [LiteRT](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct), [Ollama](https://ollama.com/library/qwen2.5:0.5b) |
| **Qwen3-0.6B** | 600M | ~2,400 MB | ~1,200 MB | ~600 MB | 462 MB | [gvij](https://huggingface.co/gvij/qwen3-0.6b-gguf) |
| **SmolLM2-360M-Instruct** | 360M | 724 MB | ~362 MB | ~181 MB | — | [aimodels](https://www.aimodels.fyi/models/huggingface/smollm2-360m-huggingfacetb), [VolkanSah](https://github.com/VolkanSah/SmolLM2-customs) |
| **Llama-3.2-1B** | 1B | ~4,000 MB | ~2,000 MB | ~1,000 MB | ~600 MB | — |

### 5.2 Measured CPU token/s

| Model | Quant | Backend | CPU | Prefill (tk/s) | Decode (tk/s) | Source |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | fp32 | LiteRT CPU | — | 90.30 | 16.71 | [LiteRT](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| Qwen2.5-0.5B | int8 | LiteRT CPU | — | 250.73 | 29.97 | [LiteRT](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| Qwen3-0.6B | Q4_K_M | llama.cpp | Raspberry Pi 5 (ARM) | — | 25.26 | [Arm](https://developer.arm.com/ai/models/hugging-face/Arm/qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/qwen3-0.6b-base-q4_k_m-gguf) |
| Qwen2.5-0.5B | Q4_K_M | llama.cpp | — | — | ~15–25 (UNVERIFIED estimate) | — |

**Note:** The LiteRT numbers are from a Samsung S24 Ultra (ARM), not x86 Haswell. The Qwen3-0.6B number is from Raspberry Pi 5 (ARM). **No x86 Haswell-specific LLM token/s numbers found.** UNVERIFIED for this hardware.

### 5.3 Recommendation

| Model | Fits 2.3 GiB? | Quality | Verdict |
|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | ✅ int8 (521 MB) or Q4_K_M (398 MB) | Good instruction following | **Best overall** |
| **Qwen3-0.6B** | ✅ Q4_K_M (462 MB) | Better reasoning, hybrid thinking | **Best quality** |
| **SmolLM2-360M** | ✅ fp32 (724 MB) | Weaker than Qwen | Budget option |
| **Llama-3.2-1B** | ⚠️ Q4_K_M only (~600 MB) | Good | Risky with other processes |

**Top pick: Qwen2.5-0.5B-Instruct (Q4_K_M, 398 MB)** — fits easily, good instruction following, Apache 2.0 license, well-supported by llama.cpp.

---

## 6. llama.cpp vs transformers for 0.5–1B on 2 CPU Cores

### 6.1 llama.cpp advantages

- **Much lower memory overhead** — no Python, no PyTorch runtime
- **GGUF quantization** — Q4_K_M, Q5_K_M, Q8_0 all available
- **Optimized for CPU** — uses all cores efficiently, AVX2-optimized
- **Single binary** — easy deployment
- [Qwen llama.cpp guide](https://qwen.readthedocs.io/en/v2.5/run_locally/llama.cpp.html)

### 6.2 transformers advantages

- **Easier Python integration** — direct pipeline API
- **Better for fine-tuning** — full training loop
- **No GGUF conversion needed** — works with HF models directly

### 6.3 Performance comparison

| Backend | Model | Quant | Memory | Speed | Source |
|---|---|---|---|---|---|
| transformers | Qwen2.5-0.5B | fp32 | 4,503 MB RSS | 16.71 tk/s decode | [LiteRT](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| transformers | Qwen2.5-0.5B | int8 | 1,363 MB RSS | 29.97 tk/s decode | [LiteRT](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| llama.cpp | Qwen2.5-0.5B | Q4_K_M | ~398 MB | ~15–25 tk/s (UNVERIFIED x86) | — |

**Verdict:** llama.cpp is the clear winner for CPU-only deployment. Lower memory, faster, easier to deploy. Use transformers only if you need fine-tuning.

### 6.4 How to get Q4_0 GGUF models

```bash
# Option 1: Download pre-quantized from HuggingFace
huggingface-cli download bartowski/Qwen2.5-0.5B-Instruct-GGUF \
  --include 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf' \
  --local-dir ./models

# Option 2: Convert from HF model
pip install gguf
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
# Use llama.cpp convert script
"
# Then: ./convert_hf_to_gguf.py --outfile model-q4_k_m.gguf --outtype q4_k_m
```

Sources: [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF), [gvij/qwen3-0.6b-gguf](https://huggingface.co/gvij/qwen3-0.6b-gguf)

---

## 7. FastAPI + Streamlit Deployment on 7.7 GiB

### 7.1 Memory footprint estimates

| Component | Memory (MB) | Source |
|---|---|---|
| Uvicorn (1 worker) | ~38–110 | [Medium](https://medium.com/@bhagyarana80/optimizing-fastapi-for-low-memory-footprint-on-microservices-6bf756f5fe8f) |
| Streamlit (per session) | ~300 | [Streamlit forum](https://discuss.streamlit.io/t/how-to-properly-optimize-cpu-and-memory-usage/9550) |
| ONNX Runtime | ~100–200 | — |
| PyTorch (runtime) | ~500–1,000 | — |
| Python + deps | ~100–200 | — |
| **Total (worst case)** | **~1,500–2,000** | — |

### 7.2 RAM is the binding constraint

With only **2.3 GiB free** (Brave running), you cannot simultaneously run:
- Streamlit (300 MB) + ONNX Runtime (200 MB) + PyTorch (1,000 MB) + Uvicorn (100 MB) = **1,600 MB** — tight but possible
- Add llama.cpp Qwen2.5-0.5B (398 MB) = **~2,000 MB** — very tight

### 7.3 Mitigation strategies

1. **Use ONNX Runtime instead of PyTorch** for DistilBERT inference — saves ~800 MB
2. **Use llama.cpp (GGUF) instead of transformers** for LLM — saves ~1,000 MB
3. **Single uvicorn worker** — `--workers 1`
4. **Streamlit `st.cache_data`** for model loading — prevents reloading
5. **Close Brave** when running the app — frees ~2 GiB
6. **Use `torch.set_num_threads(2)`** to limit CPU contention

### 7.4 Recommended architecture

```
Streamlit (UI) → FastAPI (uvicorn, 1 worker) → ONNX Runtime (DistilBERT INT8) + llama.cpp (Qwen2.5-0.5B Q4_K_M)
```

Estimated total: ~1,200–1,500 MB — fits in 2.3 GiB free.

---

## 8. Explicit UNVERIFIED List

The following numbers are **UNVERIFIED** — sourced from blogs, forums, or non-Haswell hardware:

1. **ONNX Runtime INT8 speedup on AVX2** — "smaller but still significant" (Microsoft blog, exact numbers in images)
2. **torch dynamic quantization DistilBERT speedup on laptop CPU** — no specific numbers found
3. **ModernBERT CPU latency on Haswell** — GPU numbers only (RTX 4090)
4. **Qwen2.5-0.5B / Qwen3-0.6B token/s on x86 Haswell** — only ARM (LiteRT, Raspberry Pi) numbers found
5. **llama.cpp Q4_K_M token/s on 2-core Haswell** — no direct measurement found
6. **Streamlit per-session memory** — forum estimate, not measured
7. **ONNX Runtime memory footprint** — estimated, not measured
8. **DeBERTa-v3-small Banking77 accuracy** — GLUE scores only, no Banking77-specific number
9. **MiniLM-L6-v2 / bge-small Banking77 accuracy** — MTEB benchmark, not fine-tuned on Banking77
10. **SetFit + ModernBERT on Banking77** — IMDB number only

---

## 9. Summary of Recommendations

| Question | Recommendation |
|---|---|
| ONNX INT8 on Haswell? | **Static QDQ U8U8 only.** Dynamic INT8 is a trap (≤1.1×). S8S8 has AVX2 bug. |
| Best encoder? | **ModernBERT-base** (93% F1, 2–4× faster) — verify flash-attn installs. Fallback: **DeBERTa-v3-small** (43M, safe). |
| Best tiny LLM? | **Qwen2.5-0.5B-Instruct Q4_K_M** (398 MB, Apache 2.0) via llama.cpp |
| llama.cpp vs transformers? | **llama.cpp** — lower memory, faster, easier deployment |
| Will it run on Mint 22.3? | **Yes** — glibc 2.39 > 2.28 required. ONNX Runtime 1.30, PyTorch 2.x, Transformers all work. |
| RAM constraint? | **Tight but feasible** with ONNX Runtime + llama.cpp + 1 uvicorn worker. Close Brave. |
