# VRAM Budget Forecaster

A ComfyUI custom node pack that learns, run after run, to predict the **peak
VRAM/RAM** of a workflow *before* you launch it — from real data measured on
your own machine.

**Approach: empirical, not analytical.** We don't try to compute VRAM by
reading the graph — heterogeneous custom nodes, variable attention backends,
dynamic quantization and block swap make that hopeless. Instead we measure the
real peaks (`torch.cuda.max_memory_allocated()`) on every run, store them, and
train a classic regression model on top. An LLM layer (later phase) only
*explains* the number; it never computes it.

---

## Status

| Phase | What | State |
|-------|------|-------|
| 1 | Local logging (SQLite) | **In progress** — `core/db.py`, `core/env_detect.py`, `core/monitor.py`, `core/session.py`, and the two logger nodes are done |
| 2 | Regression model (`scikit-learn`) | not started |
| 3 | Explanation layer (Gemma 3 via Ollama) | not started |
| 4 | Community data export | schema-ready, not built |

---

## Phase 1 — how to log your runs

The pack installs two nodes under the **VRAM Forecaster** category:

- **VRAM Forecaster — Logger Start**
- **VRAM Forecaster — Logger End**

### Wiring

Both nodes have a wildcard `passthrough` socket, so you splice them into an
existing wire without changing data flow:

```
[loaders] ──▶ Logger Start ──▶ [KSampler / WanVideo sampler] ──▶ Logger End ──▶ [SaveImage/Video]
                   │                                                  ▲
                   └────────────────── session ──────────────────────┘
```

1. Drop **Logger Start** just before the region you want to measure (usually
   right before the sampler). Route the latent/model wire *through* it.
2. Drop **Logger End** just after that region, before your save node. Route the
   output wire *through* it.
3. **Wire `Logger Start.session` → `Logger End.session`.** This is what forces
   End to run after the work it measures.

### What you declare on Logger Start

The model can only learn from settings it can see, so you declare the ones that
matter for your workflow as node widgets:

- resolution (w × h), frame count, loop iteration count
- block swap count, LoRA count, total LoRA weight (MB)
- optionally: main model name, quantization type (fp16/fp8/gguf/…), model size

Everything about the *environment* — GPU model, total VRAM, driver, CUDA,
PyTorch, ComfyUI version, the **active attention backend**, and installed node
pack versions — is detected automatically at runtime.

### What gets measured

- **VRAM peak**: `torch.cuda.max_memory_allocated()`, reset at Logger Start and
  read at Logger End (exact, zero-overhead high-water mark).
- **RAM peak**: a background thread samples `psutil.virtual_memory().used` every
  250 ms and keeps the maximum.
- duration, and the outcome (`ok` / `error`).

Data is written to `data/forecaster.db` (SQLite, gitignored).

---

## How OOM / crash runs are captured

OOM is the single most valuable signal — it marks the ceiling — but a crash
never reaches Logger End. Three layers handle this (see `core/session.py`):

1. **Normal close** — Logger End writes the run with the exact peaks.
2. **atexit flush** — if the process dies mid-run, an `atexit` handler marks
   every still-open session as `error`.
3. **Stale reaping** — on the next run, Logger Start calls
   `reap_stale_runs()`, which reclassifies abandoned `running` rows (older than
   6 h, scoped to the same GPU so concurrent dual-GPU jobs aren't clobbered).

Layers 2–3 lose the exact peak but preserve the negative data point.

---

## Design notes / known pitfalls (by design, not bugs)

- **No node declares its memory use** → we measure, never read metadata.
- **Disk size ≠ VRAM size** (fp8/GGUF/LoRA change everything at load) → size is
  an optional declared hint, not a prediction input we trust.
- **Attention backend matters** (vanilla/flash/sage/xformers) → detected
  explicitly at runtime from `comfy.model_management`, never assumed.
- **Peaks are about timing**, not a static sum → we measure the real high-water
  mark over the whole measured region.
- **Per-GPU separation** → runs are tagged with GPU model; the RTX 4090 and the
  dual RTX PRO 6000 are never mixed blindly.
- **Ambient noise** (drivers, other apps, PyTorch version) → treated as residual
  model error, logged (versions captured) rather than eliminated.

---

## Privacy

No paths, prompts, or filenames beyond an optional model name are stored. The
schema separates environment/hardware data from anything sensitive, and
`db.export_anonymized()` emits numeric/categorical fields only — ready for a
future opt-in community dataset without a migration.

---

## Development

```bash
pip install pytest
python -m pytest tests/ -q
```

All core logic is testable without ComfyUI, a GPU, torch, or psutil — the
detection and measurement layers degrade gracefully when those are absent.

```
vram-forecaster/
├── __init__.py            # ComfyUI node registration
├── nodes/
│   ├── common.py          # wildcard type + shared constants
│   ├── logger_start.py    # opens a measurement session
│   └── logger_end.py      # closes it, records peaks
├── core/
│   ├── db.py              # SQLite schema + access
│   ├── env_detect.py      # runtime GPU/driver/attention detection
│   ├── monitor.py         # VRAM/RAM peak measurement
│   └── session.py         # run lifecycle + crash recovery
├── data/forecaster.db     # local DB (gitignored)
└── tests/
```
