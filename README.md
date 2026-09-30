# SageAttention
<!-- We are continuously updating more features. You could **Star** and **Watch** our repository to stay updated.

--- -->
This repository provides the official implementation of SageAttention, SageAttention2, and SageAttention2++, which achieve surprising speedup on most GPUs without lossing accuracy across all models in a plug-and-play way.

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen  
Paper: https://arxiv.org/abs/2410.02367

**SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization**  
Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen  
Paper: https://arxiv.org/abs/2411.10958

**SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training**  
Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, Jun Zhu  
Paper: https://arxiv.org/abs/2505.11594


![Local Image](./assets/2.png)
*Note: [SageAttention2++](https://arxiv.org/pdf/2505.21136) achieves higher speed while maintaining the same accuracy performance.*

## Current Features
<!-- This is a beta release of SageAttention2. We welcome any feedback on accuracy, performance issues, bugs, feature requests, or suggestions. Please feel free to open an issue or launch a pull request! -->

+ Optmized kernels for **Ampere, Ada and Hopper GPUs.**
+ INT8 quantization and smoothing for $QK^\top$ with support for varying granularities.
+ FP8 quantization for $PV$, and FP16 accumulator for FP8/FP16 $PV$.
+ Two-level accumulation strategy for $PV$ to improve accuracy in FP8 MMA and WGMMA.
+ Support `torch.compile` with non-cudagraphs mode and distributed inference.


## Project Updates
- [2025-09-27]: 🎉 [SageAttention3](https://arxiv.org/abs/2505.11594) is accepted by NeurIPS 2025 as a **Spotlight** paper! 
- [2025-09-27]: The code of [SageAttention3](https://arxiv.org/abs/2505.11594) is released in this repository at  [sageattention3_blackwell](./sageattention3_blackwell/). We would still greatly appreciate it if you could take a moment to fill out the Form in [Huggingface](https://huggingface.co/jt-zhang/SageAttention3). Please note that since SageAttention2 is more accurate, we still recommend using SageAttention2 for precision-sensitive applications.
- [2025-07-01]: The code of [SageAttention2++](https://arxiv.org/pdf/2505.21136) is released in this repository. We would still greatly appreciate it if you could take a moment to fill out the Form in [Huggingface](https://huggingface.co/jt-zhang/SageAttention2_plus). Thank you very much!

![Local Image](./assets/5090_sageattn2++.png)

![Local Image](./assets/4090_sageattn2++.png)

- [2025-06-19]: [Sparse SageAttention1 API](https://github.com/jt-zhang/Sparse_SageAttention_API) and [Sparse SageAttention2 API](https://github.com/thu-ml/SpargeAttn) can compute attention with any block sparse pattern very fast.
- [2025-05-02]: 🎉SageAttention2 and [SpargeAttn](https://github.com/thu-ml/SpargeAttn) are accepted by ICML 2025! 
- [2025-02-25]: 🔥 We release [SpargeAttn](https://github.com/thu-ml/SpargeAttn), a sparse attention based on SageAttention2, which could acclerate any model without training.
- [2025-02-15]: 🔥 The compilation code is updated to support RTX5090! On RTX5090, SageAttention reaches 560T, 2.7x faster than FlashAttention2!
- [2025-01-28]: 🔥⚡SageAttention is now available on Hopper GPUs (H100, H800, H20)! It matches the speed of FlashAttention3-FP8 but offers **much better accuracy!**

| **FlashAttention2** | **FlashAttention3** | **FlashAttention3-FP8** | **SageAttention** |
|----------------------|----------------------|----------------------|----------------------|
| ![FlashAttention2](assets/cogvideox1.5_fa2_example.gif) | ![FlashAttention3](assets/cogvideox1.5_fa3_example.gif)  | ![FlashAttention3-FP8](assets/cogvideox1.5_fa3fp8_example.gif) | ![SageAttention](assets/cogvideox1.5_sage_example.gif) |
| **25'34''** | **17'32''** | **12'14''** | **12'07''** |

*Results for [CogVideoX1.5-5B](https://huggingface.co/THUDM/CogVideoX1.5-5B) on NVIDIA H20 GPU*

![Local Image](./assets/H100_hd128.png)

![Local Image](./assets/H20_hd128.png)

- [2025-01-24]: 🎉SageAttention is accepted by ICLR 2025! 
- [2024-12-20]: 🔥Update the [SageAttention2 Paper](https://arxiv.org/abs/2411.10958).   

- [2024-12-20]: 🔥Release SageAttention 2.0.1 Beta! In this version, we introduce a new feature: per-thread quantization, which offers finer granularity while maintaining hardware efficiency.
- [2024-11-21]: 🔥SageAttention 2.0.0 beta is released! Now SageAttention has measured speedup on L20, L40, A100, A800, and A6000, RTX3090 and RTX4090.
- [2024-11-12]: Support for `sageattn_varlen` is available now.
- [2024-11-11]: Support for different sequence lengths between `q` and `k,v`,  `(batch_size, head_num, seq_len, head_dim)` or `(batch_size, seq_len, head_num, head_dim)` input shapes, and `group-query attention` is available now.


## Installation
### Base environment
+ `python>=3.9`   , `torch>=2.3.0`  , `triton>=3.0.0` 
- `CUDA`:
  + `>=12.8` for Blackwell or SageAttention2++
  + `>=12.4` for fp8 support on Ada
  + `>=12.3` for fp8 support on Hopper
  + `>=12.0` for Ampere
+ `flash-attn` for benchmarking

### Install Package

For SageAttention V1 in Triton (slower than SageAttention V2/V2++/V3), refer to [SageAttention-1](https://github.com/thu-ml/SageAttention/tree/sageattention-1) branch and install using pip: `pip install sageattention==1.0.6`

To use SageAttention 2.2.0 (containing SageAttention2++), you can install using pip:
```
pip install sageattention==2.2.0 --no-build-isolation
```

**Or** you can compile from source:
```
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
python setup.py install
```

To benchmark the speed against FlashAttention3, please compile FlashAttention3 from source:
```
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release
cd hopper
python setup.py install
```

## How to Use
```python
from sageattention import sageattn
attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```
+ `q, k, v` are **FP16/BF16** dtype with the shape `(batch_size, head_num, seq_len, head_dim)` using default `tensor_layout="HND"`. For shape `(batch_size, seq_len, head_num, head_dim)`, set `tensor_layout="NHD"`. 
+ `is_causal` determines the use of a causal mask.

### Available APIs:
+ `sageattn`: Automatically selects the optimal kernel based on the GPU to achieve a good performance-accuracy trade-off.
+ `sageattn_qk_int8_pv_fp16_triton`: INT8 quantization for $QK^\top$ and FP16 for $PV$ using Triton backend.
+ `sageattn_qk_int8_pv_fp16_cuda`: INT8 quantization for $QK^\top$ and FP16 for $PV$ using CUDA backend.
+ `sageattn_qk_int8_pv_fp8_cuda`: INT8 quantization for $QK^\top$ and FP8 for $PV$ using CUDA backend. (Note that setting `pv_accum_dtype=fp32+fp16` corresponds to SageAttention2++.)
+ `sageattn_qk_int8_pv_fp8_cuda_sm90`: INT8 quantization for $QK^\top$ and FP8 for $PV$ using CUDA backend, specifically optimized for Hopper GPUs.
+ `sageattn_varlen`: INT8 quantization for $QK^\top$ and FP16 for $PV$ using Triton backend. Support for varying sequence lengths within the same batch.

For optimal speed and accuracy performance on custom devices and models, we strongly recommend referring to the [this file](./sageattention/core.py) for detailed guidance.

> **Note:**
Support for different sequence lengths between `q` and `k,v` and `group-query attention` is available.


### Plug-and-play Example

We can replace `scaled_dot_product_attention` easily. 
We will take [CogvideoX](https://huggingface.co/zai-org/CogVideoX-2b) as an example:

Add the following codes and run
```diff
import torch.nn.functional as F

+ from sageattention import sageattn
+ F.scaled_dot_product_attention = sageattn

```

Specifically,

```bash
cd example
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage
```

**You can get a lossless video in** `./example/videos/<model>/<attention_type>/` **faster than by using** `--attention_type sdpa`. More examples and guidance can be found under the `example/` directory.

> **Note:** Not all models works with `F.scaled_dot_product_attention = sageattn`. Technically, you should replace the original Attention by modifying the `Attention Class` of the target model. For image and video models, we suggest only replacing the attention in DiT (see `example/modify_mochi.py` for detail).

### Kernel Benchmarking
We provide a benchmarking script to compare the speed of different kernels including SageAttention, FlashAttention2 and FlashAttention3. Please refer to the `benchmark/` directory for more details.
 
## Performance
### Speed of Kernels

`8+8` means the kernel with INT8 quantization for $QK^\top$ and FP8 quantization for $PV$. `8+16` uses FP16 with FP16 accumulator for $PV$.

![Local Image](./assets/5090_sageattn2++.png)

![Local Image](./assets/4090_sageattn2++.png)

![Local Image](./assets/4090_hd128.png)

![Local Image](./assets/L20_hd128.png)

![Local Image](./assets/H100_hd128.png)

![Local Image](./assets/H20_hd128.png)

![Local Image](./assets/A100_hd128.png)

![Local Image](./assets/3090_hd128.png)

> **Note:** The TOPS results refer only to the Attention Kernel, excluding the quantization and smoothing.

### End-to-end Performance
#### **End-to-End Accuracy:**

![Local Image](./assets/22.png)

![Local Image](./assets/23.png)

![Local Image](./assets/24.png)

![Local Image](./assets/25.png)

#### **End-to-End Speedup:**

![Local Image](./assets/26.png)
*Note: SageAttention2++ achieves higher speed.*

## Citation
**If you use this code or find our work valuable, please cite:**
```
@inproceedings{zhang2025sageattention,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
@inproceedings{zhang2024sageattention2,
  title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization},
  author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
@article{zhang2025sageattention2++,
  title={Sageattention2++: A more efficient implementation of sageattention2},
  author={Zhang, Jintao and Xu, Xiaoming and Wei, Jia and Huang, Haofeng and Zhang, Pengle and Xiang, Chendong and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.21136},
  year={2025}
}
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```

---

## Local Setup Log (2x RTX 4080 SUPER, 2026-09-30)

Steps used to build and run this repo from scratch on a machine with 2x RTX 4080 SUPER
(sm_89, 16 GB each), Ubuntu 24.04, driver 580.65.06. The system CUDA toolkit is 12.0, which
is too old (FP8 on Ada needs >= 12.4, SageAttention2++ needs >= 12.8), so CUDA 12.8 is
installed via conda-forge inside a mamba environment.

### 1. Create the environment (mamba)

```bash
export MAMBA_ROOT_PREFIX=$HOME/.local/share/mamba
mamba create -y -p $MAMBA_ROOT_PREFIX/envs/sageattention -c conda-forge python=3.10
mamba install -y -p $MAMBA_ROOT_PREFIX/envs/sageattention -c conda-forge -c nvidia cuda-toolkit=12.8
```

After this, `<prefix>/bin/nvcc` is 12.8.93. All later commands use
`P=$MAMBA_ROOT_PREFIX/envs/sageattention` explicitly instead of activating the env.

### 2. Install PyTorch (CUDA 12.8 build)

`download.pytorch.org` was extremely slow from this machine (~0.5 MB/s), so the torch wheel
came from the Aliyun pytorch-wheels mirror and the dependencies from the TUNA PyPI mirror:

```bash
P=$MAMBA_ROOT_PREFIX/envs/sageattention
$P/bin/pip install \
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Result: `torch 2.11.0+cu128`, `triton 3.6.0`, plus the `nvidia-*-cu12` wheels
(cudnn 9.19, nccl 2.28.9, cublas 12.8, ...). Note that installing torch also downgrades
`setuptools` to `<82`, as required by its metadata.

### 3. Install the example dependencies

```bash
$P/bin/pip install ninja numpy "diffusers>=0.35" "transformers>=4.49" accelerate safetensors \
  sentencepiece protobuf ftfy imageio imageio-ffmpeg pillow tqdm debugpy \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Build SageAttention from source (sm_89)

The conda-forge gcc 14.4 toolchain is rejected by CUDA 12.8 (torch requires a host compiler
below gcc 14), so the system gcc 13.3 is used instead:

```bash
cd SageAttention
P=$MAMBA_ROOT_PREFIX/envs/sageattention
export CUDA_HOME=$P
export PATH=$P/bin:$PATH           # env nvcc 12.8 + ninja
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4080 SUPER is sm_89
export MAX_JOBS=24
$P/bin/pip install . --no-build-isolation --no-deps
```

This installs `sageattention 2.2.0` with the `_qattn_sm80`, `_qattn_sm89` and `_fused`
extensions compiled for sm_89 (~7 min with 24 parallel jobs).

### 5. Verification (RTX 4080 SUPER, fp16, HND)

Correctness vs `F.scaled_dot_product_attention`, `(2, 8, seq, head_dim)`:

| case | max abs err |
|---|---|
| non-causal, hd 64/128, seq 1024-8192 | 0.005 - 0.020 |
| causal, hd 64/128, seq 1024-8192 | 0.10 - 0.15 (concentrated in the first rows), mean err 0.0015 |

The larger causal error is expected behavior of 8-bit PV quantization, not a bug: for causal
attention the first rows attend to only 1-2 keys, so the FP8 error on `V` cannot average out
(row 0 output is just the quantized `v[0]`). Rows later in the sequence converge to the same
accuracy as the non-causal case. Use the FP16-PV path
(`sageattn_qk_int8_pv_fp16_cuda`) if this matters for your model.

Speed (`b=4, h=32`, fp16), vs SDPA:

| head_dim | seq_len | causal | SageAttention | SDPA | speedup | TOPS |
|---|---|---|---|---|---|---|
| 64 | 4096 | no | 2.92 ms | 5.48 ms | 1.87x | 188 |
| 64 | 16384 | no | 35.96 ms | 86.66 ms | 2.41x | 245 |
| 128 | 4096 | no | 5.62 ms | 10.86 ms | 1.93x | 196 |
| 128 | 16384 | no | 66.33 ms | 171.27 ms | 2.58x | 265 |
| 128 | 16384 | yes | 38.14 ms | 88.50 ms | 2.32x | 231 |

### 6. Run the plug-and-play example (CogVideoX-2b)

The weights were already in the local HF cache (with `HF_ENDPOINT=https://hf-mirror.com`):

```bash
cd example
$P/bin/python cogvideox_infer.py --model cogvideox-2b --attention_type sage --start 0 --end 1
$P/bin/python cogvideox_infer.py --model cogvideox-2b --attention_type sdpa --start 0 --end 1
```

50 denoising steps, 49 frames, 720x480, one prompt (`--start 0 --end 1`):

| attention | s/it | wall time |
|---|---|---|
| sage | 1.89 | 119 s |
| sdpa | 2.84 | 160 s |

=> ~1.5x end-to-end speedup. Videos are written to
`example/videos/cogvideox-2b/<attention_type>/0.mp4`. Note that the outputs are not
frame-identical even with a fixed seed, because small numerical differences between the two
attention implementations compound over the sampling trajectory. `--compile` was not used for
these runs (the first compiled run would additionally pay a `max-autotune` warmup cost).

### 7. Remote debugging

```bash
cd example
$P/bin/python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
  cogvideox_infer.py --model cogvideox-2b --attention_type sage --start 0 --end 1
```

Attach from VS Code/PyCharm with type `debugpy`, request `attach`, host `<host-ip>`, port
`5678`. `--wait-for-client` blocks until the debugger is attached; drop it to only debug
after startup.
