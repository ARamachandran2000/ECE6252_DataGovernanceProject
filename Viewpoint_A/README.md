# Viewpoint A

This folder contains Viewpoint A code contributions and the generated synthetic voice datasets for Indian and Southern African accents.

## What is included

- `src/`
  - Whisper setup utility and accent-experiment helper modules.
- `utils/`
  - End-to-end utilities for:
    - dataset inspection,
    - manifest building,
    - synthetic data generation,
    - Whisper fine-tuning/evaluation,
    - low-data synthetic replacement experiments,
    - plotting/report generation.
- `datasets/synthetic_data_generation/`
  - `india_and_south_asia_india_pakistan_sri_lanka/`
  - `southern_african_south_africa_zimbabwe_namibia/`
- `download_dataset.py`
  - Dataset download helper.
- `requirements.txt`
  - Python dependencies.

## Environment setup

```bash
cd Viewpoint_A
# optional: python -m venv .venv && source .venv/bin/activate
# or activate your conda environment
pip install -r requirements.txt
```

## Common Voice location expected by utilities

By default, utilities expect Common Voice under:

`./cv-corpus-25.0-2026-03-09`

Override with `--cv-root /path/to/cv-corpus-25.0-2026-03-09` when needed.

## 1) Whisper setup sanity check

```bash
python utils/smoke_test.py
```

Optional single-file transcription:

```bash
python utils/transcribe.py /path/to/audio.wav --model-id openai/whisper-base
```

## 2) Generate subset manifests (percentage-based)

```bash
python utils/make_subset_manifests.py \
  --src-dir manifests \
  --dst-dir manifests_subset \
  --train-percent 15 \
  --dev-percent 15 \
  --test-percent 15

# Build 4-condition manifests on this subset
python utils/build_experiment_manifests.py \
  --manifests-dir manifests_subset

# Optional: direct percentage-based approximation on full Common Voice
python utils/build_experiment_manifests.py \
  --manifests-dir manifests_subset \
  --approx-train-pct 15 \
  --approx-dev-pct 15 \
  --approx-test-pct 15
```

## 3) Generate manifests for low-data replacement experiment

```bash
python utils/low_data_synth_build_manifests.py \
  --base-manifests-dir manifests_subset \
  --output-manifests-dir manifests_low_data \
  --augmented-dir datasets/augmented_low_data \
  --levels 5,10,20,50,100
```

## 4) Run training/evaluation (after manifest generation)

```bash
python utils/run_accent_experiment.py \
  --manifests-dir manifests_subset \
  --outputs-dir outputs_experiment

python utils/low_data_synth_run.py \
  --manifests-dir manifests_low_data \
  --outputs-dir outputs_low_data \
  --levels 5,10,20,50,100 \
  --fp16
```

## 5) Synthetic data generation

India/South Asia example:

```bash
python utils/synthetic_data_generation.py \
  --target-accent "India and South Asia (India, Pakistan, Sri Lanka)" \
  --voice en-IN-NeerjaNeural \
  --num-reference 120 \
  --num-candidates 80 \
  --num-keep 40 \
  --save-mel-images
```

Southern African re-selection using African English voices only:

```bash
python utils/reselect_southern_african_candidates.py
```
