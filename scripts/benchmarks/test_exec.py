import sys
import torch

print("Python version:", sys.version)
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
