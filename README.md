# CodeSwitch

XLM-R–based token-level **code-switching** prediction with two heads: **switch** (next-token language change) and **duration** (burst-length class on switch events). Train on multiple bilingual pairs from the SwitchLingua dataset.

---

## Environment setup

### 1. Create the Conda environment (one time per machine / user)

From the repository root:

```bash
conda env create -f environment.yml
conda activate cs
```

This installs Python 3.11 and all **non–PyTorch** dependencies (same pins as `requirements.txt`).

### 2. Install PyTorch (required; not included in `environment.yml`)

PyTorch must match your **GPU driver** on Linux clusters. Pick **one**:

**Linux + NVIDIA GPU (recommended for CUDA 12.x drivers, e.g. 12.0–12.8):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**macOS or CPU-only (default wheels from PyPI):**

```bash
pip install "torch>=2.2.0,<2.8.0"
```

### 3. Verify GPU (optional)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 4. Alternative: `requirements.txt` only

If you prefer not to use Conda:

```bash
python -m venv .venv && source .venv/bin/activate  # or your venv workflow
pip install torch --index-url https://download.pytorch.org/whl/cu124   # Linux GPU; adjust if needed
pip install -r requirements.txt
```

### 5. Hugging Face access

The dataset and models are on the Hugging Face Hub. Set a token when preprocessing or training anything that downloads data:

```bash
export HF_TOKEN=your_hf_token_here
```

Or pass `--hf-token` to `scripts/preprocess.py` (see script help).

---

## Running experiments

Run all commands from the **repository root** with `cs` (or your venv) activated.

### Step A — Preprocess SwitchLingua

Downloads the dataset, runs LID + label generation, and writes a pickle used by training.

```bash
python scripts/preprocess.py --output data/preprocessed.pkl
```

Defaults: all **15** language pairs, up to `max_samples_per_pair` from `codeswitch/config.py`. For a quick test:

```bash
python scripts/preprocess.py --output data/small.pkl --max-samples 500
```

### Step B — Train

```bash
python scripts/train.py \
  --data data/preprocessed.pkl \
  --checkpoint checkpoints/best_xlmr.pt \
  --results results/train_results.pkl \
  --results-json results/train_results.json
```

Useful flags (see `python scripts/train.py --help`): `--epochs`, `--batch-size`, `--train-pairs`, `--zeroshot-pairs`, `--lambda-dur`, `--freeze-encoder`, `--results-json`, etc.

The script saves the **best validation switch macro-F1** checkpoint to `--checkpoint` and writes history + per-pair metrics to `--results` (pickle). With **`--results-json`**, the **same** payload is also written as UTF-8 JSON (numpy scalars normalized for submission / diffing).

### Step C — Evaluate a saved checkpoint (optional)

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_xlmr.pt \
  --data data/preprocessed.pkl \
  --output results/eval_results.pkl \
  --results-json results/eval_results.json
```

You can use **`--results-json` alone** (no `--output`) if you only need JSON.

### Step D — Plots (optional)

```bash
python scripts/visualize.py --results results/train_results.pkl --output-dir figures/
```

---

## SLURM / cluster notes

1. Use a **GPU partition** and request at least one GPU (memory depends on `batch_size` and `max_len`).
2. Run preprocessing on a node with **internet** if the compute nodes are offline.
3. Activate the same environment inside the job, e.g.:

   ```bash
   source ~/.bashrc
   conda activate cs
   export HF_TOKEN=...
   python scripts/train.py --data "$PWD/data/preprocessed.pkl"
   ```

4. If `torch` was built for a **newer** CUDA than the job node’s driver, reinstall PyTorch for a lower CUDA wheel (e.g. `cu124`) as in the setup section.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `codeswitch/` | Library: config, data, model, trainer, evaluate, LID, visualize, `results_json` |
| `scripts/preprocess.py` | Build `data/*.pkl` |
| `scripts/train.py` | Training loop |
| `scripts/evaluate.py` | Load checkpoint, per-pair metrics |
| `scripts/visualize.py` | Figures from result pickles |
| `environment.yml` | Conda env `cs` (no PyTorch) |
| `requirements.txt` | Pip deps only (no PyTorch); comments document torch install |

---

## Syncing `environment.yml` with `requirements.txt`

PyPI packages (except PyTorch) are listed in **both** files. If you bump a version in `requirements.txt`, update the matching line under `pip:` in `environment.yml` so classmates get the same stack from `conda env create -f environment.yml`.
