# HuggingBox

HuggingBox is a Tauri desktop app for browsing Hugging Face models, generating local inference code, and running those models on your own machine.

It combines:
- a Hugging Face model browser
- per-model Python environments
- editable generated code
- a file explorer + editor workspace
- local execution with live logs
- runtime routing across `transformers`, `llama.cpp`, `diffusers`, multimodal, audio, and ONNX paths

## Download

Prebuilt releases are available here:

- [Download HuggingBox (latest release)](https://github.com/tng4480/huggingbox/releases/latest)

Release assets include with a Windows installer (`.exe`) and an `.msi` package.

If there is no release yet for your platform, build it locally using the instructions below.

## What It Does

HuggingBox is designed as a local model workbench rather than a simple model launcher.

You can:
- browse models from `huggingface.co`
- inspect model details, size, format, and hardware fit
- generate starter inference code
- edit the code directly in the built-in workspace
- run the visible editor file inside the selected model environment
- inspect stdout, stderr, and structured outputs
- manage model files and Python environments per model

Supported capability areas in the current codebase include:
- text generation and common NLP pipelines
- vision pipelines
- audio pipelines
- diffusion image generation
- multimodal/image-document understanding
- ONNX runtime for supported same-repo ONNX models

## Stack

- Tauri 2
- React 19
- TypeScript
- Zustand
- Rust
- Python model runtimes

## Project Structure

```text
src/                 React UI, hooks, services, Zustand store
src-tauri/           Rust backend, Tauri commands, bundling config
hf_auto_runner/      Python runtime/router/script generation logic
docs/                PRD, design spec, sprint plan
```

## Requirements

### To run HuggingBox on Windows

- Windows 10 or 11
- WebView2 runtime
- internet access for browsing/downloading models
- enough disk space for model files and Python environments

Optional but recommended:
- a Hugging Face token for gated/private models
- an Anthropic API key if you want to use Claude Sonnet code generation or Fix mode
- a CUDA-capable GPU for faster local inference on supported models

Notes:
- HuggingBox manages per-model Python environments itself.
- Some models may still require extra Python packages or system tools depending on their runtime.
- Audio/document workflows may rely on bundled or system `ffmpeg` depending on your setup.

### To build HuggingBox from source

- Node.js 18+
- Rust toolchain
- Tauri prerequisites
- Python 3.11+

## Development

Install dependencies:

```powershell
npm install
```

Run the desktop app in development mode:

```powershell
npm run tauri dev
```

## Build

Validate the app first:

```powershell
cmd /c npx tsc --noEmit
cargo check --manifest-path src-tauri/Cargo.toml
python -m py_compile hf_auto_runner/script_generator.py hf_auto_runner/runtime_router.py hf_auto_runner/dependency_manager.py
```

Create a production build:

```powershell
npm run tauri build
```

Build artifacts will be placed under:

- `src-tauri/target/release/`
- `src-tauri/target/release/bundle/`

## Current Platform Status

- Windows: primary supported development/build target
- Linux: feasible, but still requires cross-platform hardening/testing
- macOS: planned, but not a trivial build target from Windows

## Runtime Notes

HuggingBox does not run a single fixed backend. It routes models into the most appropriate local runtime based on format, architecture, and pipeline.

Examples:
- GGUF LLMs -> `llama.cpp`
- supported ONNX models -> `onnxruntime`
- diffusion models -> `diffusers`
- multimodal document/image models -> multimodal transformers path
- audio models -> audio transformers path

Execution runs the code currently visible in the editor, not a hidden internal script.

## Settings and Environments

HuggingBox creates isolated Python environments per model by default. On first run, it can:

- install download/runtime dependencies
- download model files
- run compatibility/dependency probes

After first run, dependency management is intentionally more manual so users can control the environment through the built-in terminal.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Important:
- This license covers the HuggingBox application code in this repository.
- A [NOTICE](NOTICE) file is included for attribution handling in downstream redistributions.
- Hugging Face models, downloaded weights, and third-party runtime dependencies may be licensed separately.
- Always check the license terms for any model or dependency you use or redistribute.

## Reliable preparation and repeat runs

Open a model workspace, enter an input, and choose Run. Preparation verifies the selected environment and downloads the selected runtime files. A successful setup is reused until the recipe, model variant, device, or environment changes. Repair setup & retry rechecks requirements after a failure. Stop cancels preparation as well as inference.

New downloads live in `<model workspace>/.weights`. Removing a downloaded model removes only this managed cache; scripts, output files, and legacy downloads are preserved. Legacy scripts remain editable: use Re-gen to adopt the local-cache recipe contract (the existing script is backed up first). Old weights are not silently moved or deleted.

The worker keeps reusable model components in memory between runs. Unload model releases that memory. Changing the model or recipe restarts the worker. Prompts and workspace choices survive navigation, and the last 20 runs can be compared/exported. Output previews are limited to 16,000 characters; generated files stay in the workspace.

## Signed-in coding assistants

Settings → Code Generation offers **Codex CLI**, **Claude Code**, **Auto Runner**, and **Claude Sonnet API**. Install and sign in to your chosen CLI first (`codex login` or launch `claude`). HuggingBox uses that CLI login; it does not require an API key for CLI providers. Account access and usage limits still apply. Gated Hugging Face repositories still require their own access token.

CLI executables are detected from PATH and common install locations; an absolute executable path can be supplied in Settings. Requests send model documentation and hardware context, plus the current script and diagnostic logs for repair. Stored HuggingBox tokens are redacted from those prompts. CLI responses must contain valid code/analysis/dependency JSON. They return code into the editor; they do not automatically run generated inference. Requests time out after three minutes. Claude Code is invoked without tools; Codex uses a read-only sandbox in a separate temporary working directory.

Official CLI documentation: https://learn.chatgpt.com/docs/non-interactive-mode and https://code.claude.com/docs/en/headless.

## Regression checks

```sh
npm test
npm run test:python
cargo test --manifest-path src-tauri/Cargo.toml --lib
npm run build
npm run tauri build
```

The repair was tested on macOS with a real offline Qwen2.5-0.5B-Instruct run and a second input reusing the loaded model. Windows installers and CUDA/DirectML execution still need testing on those platforms. Runtime routing is not a guarantee that every Hub architecture works; unsupported model/task combinations fail explicitly.

## Model Lab

The Lab tab supports local LoRA training, held-out base/adapter comparisons, and exploratory ordinary/J-lens inspection. See [Model Lab](docs/model-lab.md) for supported backends, data format, checkpoint recovery and important interpretation limits.
