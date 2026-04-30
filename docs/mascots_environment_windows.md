# MASCOTS Windows GPU Environment

This repository uses a dedicated Conda prefix environment for the MASCOTS
integration:

```text
<repo-drive>:\.conda_envs\mascots310
```

The short path avoids Windows long-path failures when installing TensorFlow.
The environment is intentionally outside the repository tree but on the same
drive as the repository.

## Prerequisites

- Windows.
- NVIDIA GPU driver installed.
- Miniconda or Anaconda available from Anaconda Prompt / Miniconda Prompt.
- Repository cloned on a drive with enough free space.
- Long paths enabled is recommended, but the short environment path avoids the
  TensorFlow install issue we hit previously.

## Create The Environment

From Anaconda Prompt or Miniconda Prompt:

```bat
cd /d D:\Users\mrefoyo\Proyectos\Sub-SpaCE_plus
scripts\create_mascots_tf_gpu_py310_env.bat
```

If the repository is on another drive/path, change only the `cd` command. The
script automatically creates the environment on that same drive:

```text
<drive>:\.conda_envs\mascots310
```

The script installs:

- Python 3.10
- Conda `cudatoolkit=11.2` and `cudnn=8.1.0`
- TensorFlow 2.10.1 GPU stack
- NumPy 1.23.5
- BoRF/MASCOTS dependencies: scipy, pandas, scikit-learn, numba 0.60, llvmlite
  0.43, shap 0.42.1, loguru, psutil, awkward, sparse, etc.
- CUDA-enabled PyTorch `2.4.1+cu124`
- PyTorch Lightning `2.5.0.post0`

At the end it verifies TensorFlow and Torch in the same process:

```text
tf 2.10.1 [PhysicalDevice(...GPU...)]
torch 2.4.1+cu124 True <GPU name>
```

## Activate

```bat
conda activate "D:\.conda_envs\mascots310"
```

Use the correct drive letter if the repository is not on `D:`.

## Run The Smoke Experiment

```bat
scripts\run_mascots_experiment.bat
```

This runs:

```bat
python -m experiments.mascots
```

using the same prefix environment.

## Repair Scripts

If the environment already exists and only dependencies need repair:

```bat
scripts\install_mascots_patch_deps.bat
scripts\install_mascots_torch_deps.bat
```

## Notes

- Native Windows TensorFlow GPU support requires TensorFlow 2.10.x. Do not
  upgrade TensorFlow in this environment.
- The explained model is TensorFlow. The MASCOTS surrogate is the original
  PyTorch MLP and runs on CUDA when available.
- TensorFlow memory growth is enabled in `experiments/mascots.py` to allow
  TensorFlow and PyTorch to share the GPU.
- Cached MASCOTS builds are stored under:

```text
experiments/models/<dataset>/<model_to_explain>/mascot_build/
```
