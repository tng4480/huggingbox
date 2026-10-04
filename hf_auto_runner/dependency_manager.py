import os
import subprocess
import sys
from typing import List

HF_TRANSFORMERS_GIT_URL = "git+https://github.com/huggingface/transformers.git"

# PyTorch CUDA 12.1 index — CUDA-linked wheels for torch/torchvision/torchaudio.
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"

# llama-cpp-python pre-built wheel indexes (abetlen's GitHub Pages mirror).
LLAMA_CPP_CUDA_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cu121"
LLAMA_CPP_VULKAN_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/vulkan"

_TORCH_PKGS = {"torch", "torchvision", "torchaudio"}

# Runtimes that use the PyTorch device stack (not the C++ backends).
_PYTORCH_RUNTIMES = {
    "diffusers",
    "transformers_llm",
    "transformers_audio",
    "transformers_multimodal",
    "transformers_generic",
}


class DependencyManager:
    def __init__(self, python_exec: str, runtime: str, gpu_backend: str = "cpu"):
        self.python_exec = python_exec
        self.runtime = runtime
        self.gpu_backend = gpu_backend

    def get_dependencies(self) -> List[str]:
        if self.runtime == "llama_cpp":
            return ["llama-cpp-python", "huggingface_hub"]

        if self.runtime == "diffusers":
            return ["diffusers", "transformers", "accelerate", "torch", "torchvision", "pillow"]

        if self.runtime == "onnxruntime":
            return ["optimum", "onnxruntime", "transformers", "torch", "pillow"]

        if self.runtime == "transformers_llm":
            return ["transformers", "accelerate", "torch", "sentencepiece"]

        if self.runtime == "transformers_multimodal":
            return [
                "transformers",
                "accelerate",
                "torch",
                "torchvision",
                "pillow",
                "pymupdf",
                "sentencepiece",
                "qwen-vl-utils",
                "einops",
                "requests",
                "matplotlib",
                "addict",
            ]

        if self.runtime == "transformers_audio":
            return ["transformers", "accelerate", "torch", "librosa", "soundfile", "sentencepiece"]

        # Generic transformers fallback
        return ["transformers", "accelerate", "torch", "sentencepiece", "pillow"]

    def install_dependencies(self):
        # ── llama_cpp: vendor-specific wheel ──────────────────────────────────
        if self.runtime == "llama_cpp":
            self._install_llama_cpp()
            self._run_pip(["huggingface_hub"])
            return

        # ── diffusers on AMD/Intel: use stable-diffusion-cpp-python (Vulkan) ─
        if self.runtime == "diffusers" and self.gpu_backend in ("amd", "intel"):
            self._install_sd_cpp()
            self._run_pip(["huggingface_hub", "pillow"])
            return

        deps = self.get_dependencies()

        # ── onnxruntime → onnxruntime-directml on non-CUDA ────────────────────
        if self.runtime == "onnxruntime" and self.gpu_backend != "cuda":
            deps = [
                "onnxruntime-directml" if d == "onnxruntime" else d
                for d in deps
            ]

        # ── torch family: CUDA wheel or DirectML swap ─────────────────────────
        torch_pkgs = [d for d in deps if d in _TORCH_PKGS]
        other_deps = [d for d in deps if d not in _TORCH_PKGS]

        if torch_pkgs:
            if self.gpu_backend in ("amd", "intel") and self.runtime in _PYTORCH_RUNTIMES:
                # Replace the torch family with torch-directml (DirectX 12 backend).
                # Do NOT install standard torch alongside it — they conflict.
                print("[HuggingBox] AMD/Intel: installing torch-directml instead of torch.", flush=True)
                other_deps.append("torch-directml")
            else:
                self._install_torch_cuda(torch_pkgs)

        # ── remaining deps ────────────────────────────────────────────────────
        normalized: List[str] = []
        for dep in other_deps:
            if dep.strip().lower().startswith("transformers"):
                normalized.append(HF_TRANSFORMERS_GIT_URL)
            else:
                normalized.append(dep)
        self._run_pip(normalized)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _install_llama_cpp(self):
        if self.gpu_backend == "cuda":
            print("[HuggingBox] Installing llama-cpp-python (CUDA 12.1).", flush=True)
            try:
                self._run_pip(["llama-cpp-python"], extra_index_url=LLAMA_CPP_CUDA_INDEX_URL)
            except SystemExit:
                print(
                    "[HuggingBox] CUDA wheel unavailable for this Python version; "
                    "falling back to CPU llama-cpp-python.",
                    flush=True,
                )
                self._run_pip(["llama-cpp-python"])

        elif self.gpu_backend in ("amd", "intel"):
            print("[HuggingBox] Installing llama-cpp-python (Vulkan).", flush=True)
            # Try a pre-built Vulkan wheel first; compile from source as fallback.
            try:
                self._run_pip(["llama-cpp-python"], extra_index_url=LLAMA_CPP_VULKAN_INDEX_URL)
            except SystemExit:
                print(
                    "[HuggingBox] Vulkan pre-built wheel unavailable; compiling from source "
                    "(this may take several minutes).",
                    flush=True,
                )
                env = os.environ.copy()
                env["CMAKE_ARGS"] = "-DGGML_VULKAN=on"
                env["FORCE_CMAKE"] = "1"
                try:
                    subprocess.run(
                        [
                            self.python_exec, "-m", "pip", "install",
                            "llama-cpp-python", "--no-binary", "llama-cpp-python",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[HuggingBox] Vulkan source compile failed: {e.stderr}", flush=True)
                    sys.exit(1)

        else:
            # CPU-only — standard wheel, AVX2/AVX512 auto-detected at load time.
            print("[HuggingBox] Installing llama-cpp-python (CPU).", flush=True)
            self._run_pip(["llama-cpp-python"])

    def _install_sd_cpp(self):
        """Install stable-diffusion-cpp-python compiled with Vulkan for AMD/Intel."""
        print(
            "[HuggingBox] AMD/Intel: installing stable-diffusion-cpp-python (Vulkan). "
            "This may take several minutes.",
            flush=True,
        )
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DSD_VULKAN=ON"
        env["FORCE_CMAKE"] = "1"
        try:
            subprocess.run(
                [
                    self.python_exec, "-m", "pip", "install",
                    "stable-diffusion-cpp-python",
                    "--no-binary", "stable-diffusion-cpp-python",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            print(f"[HuggingBox] stable-diffusion-cpp-python Vulkan build failed: {e.stderr}", flush=True)
            sys.exit(1)

    def _install_torch_cuda(self, pkgs: List[str]):
        """Install the torch family from the PyTorch CUDA 12.1 index."""
        cmd = [
            self.python_exec, "-m", "pip", "install", "--upgrade",
            "--index-url", TORCH_CUDA_INDEX_URL,
        ] + pkgs
        self._exec(cmd)

    def _run_pip(self, pkgs: List[str], extra_index_url: str | None = None):
        if not pkgs:
            return
        cmd = [self.python_exec, "-m", "pip", "install", "--upgrade"] + pkgs
        if extra_index_url:
            cmd += ["--extra-index-url", extra_index_url]
        self._exec(cmd)

    def _exec(self, cmd: List[str]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e.stderr}", flush=True)
            sys.exit(1)
