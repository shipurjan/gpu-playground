# Installing CUDA (nvcc) on WSL2

This guide walks you through installing the CUDA toolkit to compile and run `.cu` files on Windows Subsystem for Linux 2 (WSL2).

**Note:** This guide is for **Debian/Ubuntu-based distributions** in WSL2 (Ubuntu 22.04, Debian, etc.). For other distributions (Fedora, Arch, etc.), check the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads) for distribution-specific instructions.

## Prerequisites

Before starting, ensure you have:

1. **Windows 11** or **Windows 10** (version 21H2 or higher)
2. **WSL2 installed** with a Linux distribution (Ubuntu 22.04 recommended)
   - Check version: `wsl --version` in PowerShell
   - If you have WSL1, upgrade to WSL2: `wsl --set-default-version 2`
3. **NVIDIA GPU** (GTX/RTX series or newer)

## Step 1: Install NVIDIA GPU Drivers on Windows

WSL2 uses the GPU drivers from your Windows host - you do **NOT** need to install drivers inside WSL2.

1. Download the latest **Game Ready Driver** or **Studio Driver** from NVIDIA:
   - https://www.nvidia.com/download/index.aspx
2. Install the driver on Windows (not in WSL2)
3. Restart your computer if prompted

**Verify GPU is accessible from WSL2:**
```bash
nvidia-smi
```

If this command works in WSL2 and shows your GPU, you're ready to proceed.

## Step 2: Install CUDA Toolkit in WSL2

Open your WSL2 terminal and run these commands:

### Remove any old CUDA installations (optional but recommended)
```bash
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
sudo apt-get autoremove
```

**Note:** If you see errors like "Unable to locate package" or "package is not installed", that's normal - it just means those packages aren't on your system. You can proceed to the next step.

### Install CUDA Toolkit 13.1
```bash
# Update package lists
sudo apt-get update

# Install prerequisites
sudo apt-get install -y wget

# Download CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download and install CUDA repository key
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update and install CUDA
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1

# Clean up the downloaded .deb file
rm cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
```

**Note about `cuda-toolkit-13-1`:** This installs CUDA 13.1 specifically and prevents automatic upgrades to 13.2, 14.0, etc. If you want automatic updates to the latest version, use `cuda-toolkit` instead (not recommended for learning).

**For other CUDA versions:** Check the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads) and select:
- Operating System: Linux
- Architecture: x86_64
- Distribution: **WSL-Ubuntu** (even if you're using Debian)
- Version: 2.0
- Installer Type: deb (local)

## Step 3: Configure PATH

Add CUDA to your PATH so you can use `nvcc` from anywhere.

### For Bash (default)
```bash
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### For Zsh
```bash
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

## Step 4: Verify Installation

Check that `nvcc` is working:

```bash
nvcc --version
```

You should see output like:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Nov__7_07:23:37_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.80
Build cuda_13.1.r13.1/compiler.36836380_0
```

## Step 5: Test with Hello World Example

Navigate to your CUDA examples folder and compile the first example:

```bash
cd ~/repos/gpu-playground/cuda
nvcc 01-hello-gpu.cu -o 01-hello-gpu
./01-hello-gpu
```

**Expected output:**
```
Hello from CPU!
Hello from GPU thread!
Back to CPU!
```

If you see this, congratulations! CUDA is working correctly.

## Troubleshooting

### `nvidia-smi` not found in WSL2
- Make sure you have WSL2 (not WSL1): `wsl --version`
- Update your NVIDIA Windows drivers to the latest version
- Ensure your Windows version supports GPU passthrough (21H2+)

### `nvcc: command not found`
- Check if CUDA is installed: `ls /usr/local/cuda-13.1/bin/nvcc`
- If the file exists, your PATH is not set correctly - repeat Step 3
- If the file doesn't exist, CUDA installation failed - repeat Step 2

### Permission denied when running compiled program
```bash
chmod +x 01-hello-gpu
./01-hello-gpu
```

### Wrong CUDA version installed
To check installed versions:
```bash
ls /usr/local/cuda*
```

To change the symlink to a different version:
```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-13.1 /usr/local/cuda
```

### GPU not detected (`cuda runtime error`)
- Run `nvidia-smi` to verify GPU is accessible
- Make sure you installed drivers on **Windows**, not inside WSL2
- Try restarting WSL2: `wsl --shutdown` in PowerShell, then reopen WSL2

### Compilation works but execution fails
If `nvcc` compiles successfully but running the program gives errors:
- Check GPU compatibility: CUDA 13.x requires compute capability 5.0+
- Verify your GPU model: `nvidia-smi --query-gpu=name,compute_cap --format=csv`

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [WSL2 CUDA Guide (Official)](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)

## Quick Command Reference

```bash
# Check CUDA version
nvcc --version

# Check GPU status
nvidia-smi

# Compile CUDA program
nvcc filename.cu -o outputname

# Run compiled program
./outputname

# Clean up compiled files
rm outputname
```
