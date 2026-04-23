# CodeSwitch

Token-level **code-switching prediction** with two anticipatory heads:
- **Switch head** — will the next token be in a different language? (binary)
- **Duration head** — if a switch occurs, how long is the burst? (3-class: small / medium / large)

Trained on all 15 language pairs from [SwitchLingua](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) with two backbone variants (XLM-R and XGLM) and two analysis experiments (burstiness and qualitative CS-type).

---

## Experiments

| Notebook | Backbone | Key Analysis |
|----------|----------|--------------|
| `notebooks/01_xglm_gpt_backbone.ipynb` | XGLM-564M (GPT decoder) | Baseline — per-pair F1, σ-universality |
| `notebooks/02_xlmr_causal_burstiness.ipynb` | XLM-R + causal mask | Multitask vs single-task recall in bursty regions |
| `notebooks/03_xlmr_causal_cstype.ipynb` | XLM-R + causal mask | Qualitative P/R/F1 breakdown by code-switch type |

---

## Environment Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate cs
```

### 2. Install PyTorch

**Linux + NVIDIA GPU (CUDA 12.x):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**macOS or CPU-only:**
```bash
pip install "torch>=2.2.0,<2.8.0"
```

### 3. Hugging Face access

```bash
export HF_TOKEN=your_hf_token_here
```

---

## Running Experiments

All commands are run from the **repository root** with the `cs` environment active.

### Recommended: run via notebooks

Open and run the notebooks in order:
```
notebooks/01_xglm_gpt_backbone.ipynb
notebooks/02_xlmr_causal_burstiness.ipynb
notebooks/03_xlmr_causal_cstype.ipynb
```

Each notebook handles preprocessing (with caching), training, evaluation, and analysis. All hyperparameters are set at the top of each notebook for easy reproduction.

---

### Alternative: CLI scripts

#### Step A — Preprocess SwitchLingua

```bash
python scripts/preprocess.py --output data/preprocessed.pkl
```

Options: `--max-samples 1000` for a quick test, `--pairs Chinese-English Hindi-English` for specific pairs.

#### Step B — Train

```bash
# XLM-R (default) — saves results/train_xlmr.json automatically
python scripts/train.py \
  --data data/preprocessed.pkl \
  --backbone xlmr \
  --checkpoint checkpoints/best_xlmr.pt

# XGLM / GPT backbone — saves results/train_xglm.json automatically
python scripts/train.py \
  --backbone xglm \
  --checkpoint checkpoints/best_xglm.pt

# Switch-only (no duration auxiliary task)
python scripts/train.py --lambda-dur 0.0 --checkpoint checkpoints/best_xlmr_st.pt
```

Key flags (see `python scripts/train.py --help`):

| Flag | Default | Description |
|------|---------|-------------|
| `--backbone` | `xlmr` | `xlmr` or `xglm` |
| `--epochs` | 16 | Training epochs |
| `--base-lr` | 1e-5 | Encoder LR |
| `--head-lr-mul` | 50 | Head LR = base × mul |
| `--warmup-ratio` | 0.1 | Fraction of steps for LR warmup |
| `--lambda-dur` | 1.0 | Duration loss weight (0 = switch-only) |
| `--max-len` | 256 | Max token length |
| `--freeze-encoder` | off | Freeze encoder, train heads only |
| `--results-json` | `results/train_<backbone>.json` | Override default JSON output path |

#### Step C — Evaluate a checkpoint

```bash
# saves results/eval_xlmr.json automatically
python scripts/evaluate.py \
  --checkpoint checkpoints/best_xlmr.pt \
  --data data/preprocessed.pkl
```

#### Step D — Burstiness analysis

Compares multitask (λ=1.0) vs single-task (λ=0.0) recall in burst vs isolated regions:

```bash
python scripts/analyze_burstiness.py \
  --checkpoint checkpoints/best_xlmr.pt \
  --st-checkpoint checkpoints/best_xlmr_st.pt \
  --data data/preprocessed.pkl \
  --output-dir results/burstiness
```

Flags: `--burst-threshold 3`, `--window-size 5`.

#### Step E — Qualitative CS-type analysis

Per-token error breakdown by code-switch type for selected pairs:

```bash
python scripts/analyze_qualitative.py \
  --checkpoint checkpoints/best_xlmr.pt \
  --data data/preprocessed.pkl \
  --pairs Korean-English German-English Chinese-English \
  --output-dir results/qualitative
```

#### Step F — Plots

```bash
python scripts/visualize.py --results results/train_xlmr.pkl --output-dir figures/
```

---

## Repository Layout

| Path | Role |
|------|------|
| `codeswitch/config.py` | Dataclasses: `ModelConfig`, `DataConfig`, `TrainConfig`, `BurstinessConfig`, `QualitativeConfig` |
| `codeswitch/model.py` | `CausalXLMRCodeSwitchPredictor`, `XGLMCodeSwitchPredictor`, `build_model()` |
| `codeswitch/data.py` | `CodeSwitchDataset`, `CodeSwitchDatasetWithMeta`, label generation, LID alignment |
| `codeswitch/trainer.py` | `train_epoch`, `make_warmup_cosine_scheduler`, class-weight helpers |
| `codeswitch/evaluate.py` | `evaluate`, `evaluate_per_pair`, `print_sigma_summary` |
| `codeswitch/burstiness.py` | Burstiness metrics and plots (Experiment 2) |
| `codeswitch/qualitative.py` | Per-token CS-type analysis and plots (Experiment 3) |
| `codeswitch/visualize.py` | `plot_universality`, `plot_training_history`, `plot_per_pair_training_curves`, `plot_grouped_f1_bars`, `plot_lr_schedule` |
| `codeswitch/lid.py` | Multi-tier Language Identification system |
| `codeswitch/results_json.py` | Numpy-safe JSON serialization |
| `scripts/preprocess.py` | Build `data/*.pkl` from SwitchLingua |
| `scripts/train.py` | Full training loop |
| `scripts/evaluate.py` | Load checkpoint → per-pair metrics |
| `scripts/analyze_burstiness.py` | Burstiness MT vs ST comparison |
| `scripts/analyze_qualitative.py` | CS-type qualitative error analysis |
| `scripts/visualize.py` | Plots from result pickles |
| `notebooks/` | Three experiment notebooks (self-contained, use library) |
| `environment.yml` | Conda env `cs` (no PyTorch) |
| `requirements.txt` | Pip deps (no PyTorch) |

---

## SLURM / Cluster Notes

1. Use a GPU partition and request at least one GPU.
2. Run preprocessing on a node with internet access.
3. Activate environment inside the job:
   ```bash
   source ~/.bashrc
   conda activate cs
   export HF_TOKEN=...
   python scripts/train.py --data "$PWD/data/preprocessed.pkl"
   ```
4. If `torch` was built for a newer CUDA than the driver, reinstall for a lower wheel (e.g. `cu124`).

---

## Hyperparameter Reference

All experiments used:

| Parameter | Value |
|-----------|-------|
| Training pairs | 15 (all SwitchLingua pairs) |
| Max samples/pair | 6,000 |
| Max sequence length | 256 tokens |
| Batch size | 32 |
| Epochs | 16 |
| Encoder LR | 1e-5 |
| Head LR | 5e-4 (50× encoder) |
| LR schedule | Linear warmup (10%) + cosine decay |
| Optimizer | AdamW (weight\_decay=0.01) |
| Duration loss weight (λ) | 1.0 (multitask) / 0.0 (switch-only) |
| Gradient clipping | 1.0 |
| Seed | 42 |
