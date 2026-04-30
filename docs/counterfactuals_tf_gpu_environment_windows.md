# Original Counterfactuals TF-GPU Environment

This document recreates the previous environment:

```text
counterfactuals_tf_gpu
```

This is the environment used for the original repository methods before the
separate MASCOTS `mascots310` environment was created.

## Important Notes

- Native Windows TensorFlow GPU support requires TensorFlow 2.10.x.
- Do not upgrade TensorFlow in this environment if you need native Windows GPU.
- The known working stack is:

```text
Python 3.8.18
TensorFlow 2.10.1
tensorflow-gpu 2.10.1
CUDA toolkit 11.2
cuDNN 8.1
```

## Option A: Robust Recreate Script

Recommended for a new PC.

From Anaconda Prompt or Miniconda Prompt:

```bat
cd /d D:\Users\mrefoyo\Proyectos\Sub-SpaCE_plus
scripts\create_counterfactuals_tf_gpu_env.bat
```

This creates a named Conda environment:

```text
counterfactuals_tf_gpu
```

It installs the core packages used by the previous experiments and verifies
TensorFlow GPU and Torch CUDA at the end.

## Option B: Faithful Export Recreate

This uses the cleaned export generated from the original environment:

```text
environment_counterfactuals_tf_gpu.yml
```

Run:

```bat
cd /d D:\Users\mrefoyo\Proyectos\Sub-SpaCE_plus
scripts\create_counterfactuals_tf_gpu_from_export.bat
```

This is closer to the original environment but may be less portable because the
export includes many exact versions and Conda build strings.

## Activate

```bat
conda activate counterfactuals_tf_gpu
```

## Verify Manually

```bat
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected TensorFlow output includes:

```text
2.10.1 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Relationship To MASCOTS Environment

This environment is separate from:

```text
<repo-drive>:\.conda_envs\mascots310
```

Use `counterfactuals_tf_gpu` for the existing repository methods. Use
`mascots310` for the adapted MASCOTS implementation.
