import torch
if torch.backends.mps.is_available():
    print("MPS is available! PyTorch can use the GPU on Apple Silicon.")
    device = torch.device("mps")
else:
    print("MPS not available. PyTorch will use CPU.")
    device = torch.device("cpu")
print(f"Using device: {device}")

