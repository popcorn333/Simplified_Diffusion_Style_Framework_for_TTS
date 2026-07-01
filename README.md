# Simplified Diffusion Style Framework for TTS

This repository contains several diffusion-style text-to-speech experiment variants. Each variant is kept in its own directory and includes its own training script, inference script, Hydra configs, model code, HiFi-GAN vocoder code, filelists, and SLURM launch scripts.

Available experiment directories:

- `score`
- `fm`
- `add`
- `heat_equation`
- `astable`
- `mask`
- `product`

The common workflow is:

1. Choose an experiment directory.
2. Check or edit its Hydra configs under `config/`.
3. Train with `train.py`.
4. Run synthesis with `inference.py`.
5. Evaluate generated WAV files with the metric scripts if needed.

## Repository Layout

Each experiment directory follows this layout:

```text
<experiment>/
  train.py                  # Hydra training entry point
  inference.py              # Hydra inference/synthesis entry point
  run.sh                    # SLURM train + inference sweep
  inference.sh              # SLURM or shell inference helper
  metrics.sh                # metric helper
  config/
    config_swp.yaml         # training sweep config
    config_eval_swp.yaml    # inference sweep config
    data/data_swp.yaml      # dataset paths for Hydra sweeps
  checkpts/
    hifigan-config.json     # HiFi-GAN config
    hifigan.pt              # Pretrained HIFI-GAN vocoder checkpoint
    grad-tts-new_0.pt       # Grad-TTS encoder checkpoint
  resources/
    filelists/              # train/dev/test filelists
    cmu_dictionary          # pronunciation dictionary
  model/
  text/
  hifi-gan/
```

## Requirements

The runnable training and inference path expects a Python environment with CUDA-enabled PyTorch and the following packages:

```bash
pip install torch torchaudio numpy scipy tqdm tensorboard hydra-core omegaconf matplotlib librosa einops six cython
```

Important fields:

- `train_filelist_path`: training filelist
- `dev_filelist_path`: validation filelist
- `test_filelist_path`: inference/test filelist
- `cmudict_path`: CMU dictionary path
- `sample_rate`: expected WAV sample rate, default `22050`

The loader asserts that each WAV file has the configured sample rate.

## Notes

- Run commands from inside the selected experiment directory.
- Use `config_swp.yaml` and `data_swp.yaml` for the documented sweep workflow.
- The tracked filelists currently contain dummy paths. Update them for your dataset location.
