# utils/check_gpu.py
"""
A utility script to check and display information about PyTorch, CUDA, and cuDNN availability.

This script is helpful for verifying the local environment setup, especially when
intending to use GPU acceleration for training neural networks. It prints:
- The installed PyTorch version.
- Whether CUDA is available to PyTorch.
- If CUDA is available:
    - The CUDA version detected by PyTorch.
    - The cuDNN version detected by PyTorch.
    - The number of available GPUs.
    - The name of the primary GPU (device 0).
    - The name of the current CUDA device being used by PyTorch.
- It also performs a simple test of moving a tensor to the GPU and back (if available)
  to confirm basic GPU operations are working.

To run this script:
    python utils/check_gpu.py
"""

import torch  # PyTorch library, essential for deep learning and GPU checks


def check_gpu_availability_and_details():
    """
    Checks for GPU availability and prints detailed information about the
    PyTorch, CUDA, and cuDNN setup. Also performs a basic GPU tensor operation test.
    """
    print("--- PyTorch and GPU Availability Check ---")

    # --- PyTorch Version ---
    print(f"PyTorch version: {torch.__version__}")

    # --- CUDA Availability ---
    # torch.cuda.is_available() returns True if PyTorch can access a CUDA-enabled GPU
    cuda_available: bool = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # --- CUDA and cuDNN Versions (if CUDA is available) ---
        # torch.version.cuda shows the CUDA version PyTorch was compiled with
        print(f"CUDA version detected by PyTorch: {torch.version.cuda}")

        # For PyTorch 1.10 and later, cuDNN version is accessible via torch.backends.cudnn.version()
        # Check if the attributes exist to avoid errors in older PyTorch versions or if cuDNN is not found.
        if (
            hasattr(torch.backends, "cudnn")
            and hasattr(torch.backends.cudnn, "version")
            and callable(torch.backends.cudnn.version)
        ):
            try:
                cudnn_version = torch.backends.cudnn.version()
                print(f"cuDNN version: {cudnn_version}")
            except Exception as e:
                print(f"Could not retrieve cuDNN version: {e}")
        else:
            print(
                "cuDNN version not readily available through torch.backends.cudnn.version() "
                "(may indicate older PyTorch or cuDNN not fully configured)."
            )

        # --- GPU Device Information ---
        # torch.cuda.device_count() returns the number of GPUs accessible to PyTorch
        num_gpus: int = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        if num_gpus > 0:
            # torch.cuda.get_device_name(0) gets the name of the first GPU (device index 0)
            try:
                print(f"GPU Name (Device 0): {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"Could not retrieve name for GPU 0: {e}")

            # torch.cuda.current_device() returns the index of the currently selected CUDA device
            try:
                current_device_idx: int = torch.cuda.current_device()
                print(f"Current CUDA device (Index): {current_device_idx}")
                print(
                    f"Current CUDA device Name: {torch.cuda.get_device_name(current_device_idx)}"
                )
            except Exception as e:
                print(f"Could not retrieve current CUDA device info: {e}")
    else:
        # Message if CUDA is not available
        print("CUDA is NOT available. PyTorch will run on CPU.")

    # --- Test Basic GPU Tensor Operation (if CUDA is available) ---
    if (
        cuda_available and num_gpus > 0
    ):  # Ensure num_gpus check for multi-GPU systems where 0 might still be an issue
        print("\n--- GPU Tensor Operation Test ---")
        try:
            # Create a sample tensor on the CPU
            tensor_cpu = torch.randn(3, 3)  # Creates a 3x3 tensor with random numbers
            print(f"Tensor on CPU: {tensor_cpu.device}, Data: \n{tensor_cpu}")

            # Move the tensor to the GPU (default CUDA device, usually 'cuda:0')
            tensor_gpu = tensor_cpu.to("cuda")
            print(f"Tensor on GPU: {tensor_gpu.device}, Data: \n{tensor_gpu}")

            # Perform a simple operation on the GPU tensor
            tensor_gpu_doubled = tensor_gpu * 2
            print(
                f"Tensor on GPU (doubled): {tensor_gpu_doubled.device}, Data: \n{tensor_gpu_doubled}"
            )

            # Move the result back to the CPU
            tensor_back_to_cpu = tensor_gpu_doubled.to("cpu")
            print(
                f"Tensor back on CPU: {tensor_back_to_cpu.device}, Data: \n{tensor_back_to_cpu}"
            )

            print(
                "\nSuccessfully created a tensor, moved it to GPU, performed an operation, and moved it back to CPU."
            )
        except Exception as e:
            # Catch any errors during the tensor test
            print(f"Error during GPU tensor test: {e}")
            print(
                "This might indicate issues with CUDA drivers, PyTorch-CUDA compatibility, or GPU hardware."
            )
    elif cuda_available and num_gpus == 0:
        print(
            "\nCUDA reported as available, but no GPUs found by PyTorch (device_count is 0). Cannot run GPU tensor test."
        )


if __name__ == "__main__":
    # This block executes when the script is run directly
    check_gpu_availability_and_details()
    print("\n--- Check Complete ---")
