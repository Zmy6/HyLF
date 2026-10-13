# HyLF: Sparse Angular Super-resolution from Hybrid Light Fields without Known Camera Parameters

Implementation of **HyLF: Sparse Angular Super-resolution from Hybrid Light Fields without Known Camera Parameters**, published in *IEEE Transactions on Circuits and Systems for Video Technology* (2026).

Paper: [doi:10.1109/TCSVT.2026.3688726](https://doi.org/10.1109/TCSVT.2026.3688726)

## Installation

```bash
conda env create -f environment.yml
conda activate hylf
```

## Data

Each scene directory must contain only a lexicographically sortable 9x9 set of 81 RGB view images. The expected layout is:

```text
data/
  HCI_new/
    bicycle/
      <81 view images>
    boxes/
  INRIA/
    Bee/
      <81 view images>
  30scene/
    Cars/
      <81 view images>
```

## Training

All scene scripts use the same interface:

```text
<script> <gpu-id> <scene> <angular-step> <dataset-directory> [hydra-overrides...]
```

For example, train HyLF on the HCI_new `bicycle` scene using the 2x2 views of a centered 7x7 light field:

```bash
bash scripts/train_hylf.sh 0 bicycle 6 HCI_new
```

The angular step determines the reconstructed LF:

| Step | input LF | Reconstructed LF |
| ---: | :---: | :---: |
| 6 | 2x2 | 7x7 |
| 7 | 2x2 | 8x8 |
| 8 | 2x2 | 9x9 |

To reproduce the paper's Table 1-3:

```bash
bash scripts/train_paper_tables.sh
```

## Evaluation

```bash
python testY.py \
  --log-root /path/to/HyLF_runs/logs \
  --dataset all \
  --angular-step 6 \
  --validation-epoch 30 \
  --output-directory /path/to/HyLF_runs/metrics/epoch_30
```

## Ablations

The repository includes the three paper-oriented alternatives retained by the original experiment code:

```bash
bash scripts/train_ablation_two_plane.sh 0 bicycle 6 HCI_new
bash scripts/train_ablation_multi_plane.sh 0 bicycle 6 HCI_new
bash scripts/train_ablation_planar_voxel.sh 0 bicycle 6 HCI_new
```

- `ablation_two_plane` replaces the learned two-plane ray parameterization with origin-direction rays and removes angular flow.
- `ablation_multi_plane` reduces the plane-intersection feature channels from 32 to 1 and removes angular flow.
- `ablation_planar_voxel` replaces the planar neural voxel decoder with an RGB MLP and removes angular flow.


## Citation

```bibtex
@article{zhao2026hylf,
  title   = {HyLF: Sparse Angular Super-resolution from Hybrid Light Fields without Known Camera Parameters},
  author  = {Zhao, Mingyuan and Yang, Da and Cong, Ruixuan and Cui, Zhenglong and Chen, Rongshan and Ke, Wei and Sheng, Hao},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year = {2026},
  volume = {36},
  number = {9},
  pages = {13090-13102},
}
```
