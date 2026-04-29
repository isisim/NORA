# NORA
Supporting code for "A multimodal 3D dataset of an oil&amp;gas Non-Operational ReplicA (NORA)

# Installation

We provide a functioning environment file, to install using conda:

```bash
conda env create -f environment.yml -n new_env_name
```
# Downsampling

The `scripts/preprocess.py` file implements the voxel downsampling described in the paper. Usage:

```bash
python scripts/preprocess.py data/nora.ply -o data/nora_downsample.ply
```