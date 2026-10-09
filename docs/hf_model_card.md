---
license: gpl-3.0
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - embodied-ai
  - jepa
  - flow-matching
  - vision-language-action
  - libero
---

# VPA (Vision-Predictor-Action) — LIBERO-10 checkpoints

Trained checkpoints of the reference implementation of
*Vision-Predictor-Action: Towards Efficient Embodied AI via Neuro-Symbolic JEPA Predictors*
([preprint, doi:10.5281/zenodo.22960919](https://doi.org/10.5281/zenodo.22960919)).
Code, training and evaluation: <https://github.com/ylee-sling/Vision-Predictor-Action>.

VPA replaces the autoregressive language-model decoder of vision-language-action models with a
latent control loop: a shared vision encoder (E_ψ), a neuro-symbolic primitive selector, a JEPA
predictor of the next latent and a flow-matching solver that generates a whole action chunk in K
steps, so one decision step takes K + 3 sequential network evaluations regardless of the chunk length.

## Files

| File | Model |
|---|---|
| `vpa_libero10_2cam.pt` | two cameras: third-person `agentview_rgb` + wrist `eye_in_hand_rgb` |
| `vpa_libero10_1cam.pt` | one camera: `agentview_rgb` (the paper's single observation frame) |
| `*.safetensors` | the same weights in the safetensors format |
| `*.json` | model configuration (`VPAConfig`) and data record |
| `libero10_pilot.json` | the evaluation results below |

Both were trained identically with the repository's `train.py` defaults (randomly initialised ViT
encoder, 128×128 frames, 50k + 50k steps, batch 64, K = 2, H = 16, seed 0) on the LIBERO-10
demonstrations; only the cameras differ.

## Results on LIBERO-10 (preliminary)

**Preliminary:** one training run per model and 20 trials per task (200 episodes per row); the 95%
interval is about ±7 points.

| Model | Success | 95% interval |
|---|---|---|
| one camera | 10.5% (21/200) | 7.0–15.5% |
| two cameras | 40.0% (80/200) | 33.5–46.9% |

Per-task rates, the execution-horizon sweep and the blindfold test are in the
[repository README](https://github.com/ylee-sling/Vision-Predictor-Action#results-on-libero-10-preliminary).

## Use

```bash
git clone https://github.com/ylee-sling/Vision-Predictor-Action && cd Vision-Predictor-Action
pip install -r requirements.txt h5py huggingface_hub
hf download ylee-sling/vpa-libero10 vpa_libero10_2cam.pt --local-dir checkpoints
```

```python
from train import load_pipeline
pipe = load_pipeline("checkpoints/vpa_libero10_2cam.pt", device="cuda")   # calibrated, ready for reset/step/act
```

Closed-loop evaluation in LIBERO (milestone goal frames come from LIBERO-10 demonstrations):

```bash
python eval.py --checkpoint checkpoints/vpa_libero10_2cam.pt --suite-name libero_10 \
    --data-root /path/to/LIBERO/datasets/libero_10 --num-trials-per-task 20
```

The first load downloads the frozen CLIP text encoder (`openai/clip-vit-base-patch32`).

## License

GPL-3.0-only, like the code. The models were trained on the
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) datasets; their terms apply as well.

## Citation

```bibtex
@misc{lee2026vpa,
  author    = {Lee, Yeonseok},
  title     = {Vision-Predictor-Action: Towards Efficient Embodied AI via Neuro-Symbolic JEPA Predictors},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22960919},
  url       = {https://doi.org/10.5281/zenodo.22960919},
  note      = {Preprint}
}
```
