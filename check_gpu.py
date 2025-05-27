import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    # For PyTorch 1.10 and later, cuDNN version is part of torch.backends.cudnn.version()
    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'version'):
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("cuDNN version not readily available through torch.backends.cudnn.version()")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is NOT available. PyTorch will run on CPU.")

# Test a tensor operation on GPU
if torch.cuda.is_available():
    try:
        tensor_cpu = torch.randn(3, 3)
        print(f"\nTensor on CPU: {tensor_cpu.device}")
        tensor_gpu = tensor_cpu.to('cuda')
        print(f"Tensor on GPU: {tensor_gpu.device}")
        print("Successfully moved tensor to GPU and back.")
    except Exception as e:
        print(f"Error during GPU tensor test: {e}")

