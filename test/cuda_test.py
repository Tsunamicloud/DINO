import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("cuDNN 版本:", torch.backends.cudnn.version())

# NVIDIA GeForce RTX 3090 Ti with CUDA capability sm_86
print("CUDA capability:", torch.cuda.get_arch_list())