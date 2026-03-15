# 🔥 Inferno.jl

> [!WARNING]
> **Status: Experimental / Work-in-Progress**
> The codebase is currently in a pre-alpha state. While it can load GGUF models, it frequently hangs or fails to generate tokens on certain Intel GPU driver versions due to known memory synchronization issues. **It is not yet production-ready.**

**Inferno.jl** is a research-oriented, Julia-native LLM inference engine specifically targeted at **Intel GPUs** and the **oneAPI** ecosystem. The project aims to provide a lean, high-performance solution for modern transformer-based models like the **Qwen 3.5** family, while explicitly handling the unique challenges of the Intel hardware stack.

---

## 🚀 Project Goals

- **Intel GPU Native**: Deep integration with `oneAPI.jl`, targeting **Intel Arc** and **Battlemage** (B580) series.
- **Custom Quantization Kernels**: Hand-optimized dequantization for **IQ2_XXS**, **Q4_0**, and **Q8_0**.
- **Self-Contained**: Native GGUF parsing and BPE tokenization without heavy external binary dependencies.
- **Hardware Stability**: Implementing robust workarounds for Intel driver quirks through custom Level Zero integration.

---

## ⚡ Current Status & Known Issues

We are currently working through significant stability hurdles related to the Intel oneAPI driver stack.

- **Driver Poisoning**: Certain kernel dispatches (especially mixed `HostBuffer`/`DeviceBuffer` operations) can "poison" the Level Zero command queue, causes subsequent operations to hang or return zeroed data. See [oneAPI.jl #458](https://github.com/JuliaGPU/oneAPI.jl/issues/458).
- **Inconsistent Token Generation**: Due to the above, the engine may load successfully but fail to yield tokens during the `generate_stream` loop.
- **Memory Synchronization**: Shared memory (`HostBuffer`) consistency is a primary blocker. We are transitioning to explicit copy-based synchronization and custom vendor-layer patches.

---

## 📦 Installation (For Developers)

Inferno is currently intended for developers and researchers.

```julia
using Pkg
Pkg.add("Inferno")
```

### Requirements
- **Hardware**: Intel Arc (Alchemist) or Battlemage GPU.
- **Driver**: Intel Level Zero drivers installed on the system.
- **Software**: Julia 1.10+ and the Intel oneAPI Base Toolkit.

---

## 🛠️ Quick Start

### Basic Usage (Experimental)

```julia
using Inferno

# Load the model (Note: may hang if driver is in a bad state)
model, tokenizer = Inferno.load_model("path/to/qwen3.5-0.8b-iq2_xxs.gguf")

# Generation attempt
stream = Inferno.generate_stream(model, tokenizer, prompt; max_tokens=50)

for token in stream
    print(token)
    flush(stdout)
end
```

### HTTP Server

```julia
# Starts an OpenAI-compatible server on port 8080
Inferno.main("path/to/model.gguf"; port=8080)
```

---

## 🏗️ Technical Architecture

- **`Loader.jl`**: Maps GGUF tensors; handles the transition from disk to GPU memory.
- **`Dequant.jl`**: High-performance oneAPI kernels for on-the-fly dequantization.
- **`Model.jl`**: Transformer implementation with optimizations for Intel's execution model.
- **`vendor/`**: A critical layer containing custom patches for **Level Zero** to bypass driver-level issues and provide stable synchronization primitives.

---

## 🤝 How to Help

The project is currently blocked by low-level memory consistency issues on Intel hardware. We welcome help from developers familiar with **oneAPI**, **Level Zero**, and **JuliaGPU** architecture. Check the issue tracker for specific driver-repro cases.

---

## 📄 License

This project is open-source. See the repository for licensing details.