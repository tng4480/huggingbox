# Model Lab

Lab adds three local workflows: adapter fine-tuning, held-out comparison, and internal vocabulary readouts. This is a first supported implementation, not a claim of universal model compatibility or automatic model improvement.

## Fine-tuning

Choose a prepared model or a local folder containing config/tokenizer/safetensors files. MLX on Apple Silicon supports LoRA on compatible normal or quantized models; PyTorch uses full-precision weights on CPU or NVIDIA CUDA. GGUF is rejected. Click Prepare isolated runtime once for the chosen backend. Advanced users can select an existing Python executable; setup never installs into that override.

Import JSONL prompt/completion pairs or chat messages whose final message is an assistant answer. Validation rejects malformed rows, empty answers, duplicate inputs and datasets under six examples. The fixed seeded split keeps test examples out of training and validation. Training checks tokenized lengths and refuses oversized examples rather than silently truncating targets. Only answer tokens contribute to the training objective.

Training creates separate adapter files, logs, input snapshots and model/dataset hashes in the application's Lab run directory. It does not edit the original weights. The UI tracks training loss, but loss is not proof of improved behavior. Stop terminates the job; existing readable checkpoints can be used to continue. Continuation restores adapter weights with a fresh optimizer, not an exact optimizer/RNG resume. It requires the same dataset, split seed, adapter rank and layer count. Incomplete MLX checkpoints are skipped in favor of an earlier readable checkpoint.

Dataset drafts are stored on disk, separately from browser storage. Other Lab choices persist with workspace drafts. Jobs survive navigation; after an app restart, unfinished jobs are shown as interrupted. There is one Lab job at a time, and inference cannot start while it runs.

## Evaluation

Choose a completed training run. Both the original model and saved adapter receive the same held-out inputs, chat template and greedy decoding limit. Results include responses, timings and whitespace-trimmed exact-match scores. These scores are suitable for constrained answers, not a general quality judgment. Test sets remain fixed, but repeated human tuning against their results can still overfit them: keep an additional external final test set for serious evaluation.

## Inspection

The first inspection backend supports compatible Transformers decoder checkpoints in PyTorch. Ordinary logit-lens readouts work without a fitted lens. Fit J-lens uses the pinned Anthropic reference implementation, two sampled layers and 2–100 user-supplied calibration prompts. The exploratory fit is checkpoint-specific and is saved with model and lens checksums. Imported lenses must include the matching HuggingBox sidecar manifest; an arbitrary external lens file is not assumed compatible.

Click layer/token cells to view top-five vocabulary scores. They are not calibrated probabilities, a thought transcript, or proof of causal influence. Comparing a PyTorch adapter against its original model uses the same lens; a base-model J-lens transferred to the adapter is explicitly marked unvalidated. MLX adapters are not automatically converted to PyTorch adapters for inspection. Activation patching and full circuit graphs are not implemented here.

## Verification

- Automated tests cover split separation/reproducibility, duplicate/invalid examples, bounded settings, exact model identity, no silent truncation, atomic result writes and checkpoint fallback.
- Real Qwen2.5-0.5B-Instruct MLX training and base/adapter held-out evaluation completed.
- First-time isolated MLX and PyTorch/J-lens runtime installation completed through the native UI; a further native Qwen training run succeeded using the clean MLX environment.
- The native app validated data, trained an adapter, compared outputs, cancelled training and preserved readable checkpoints. Continuing a cancelled checkpoint succeeded.
- PyTorch training/evaluation and J-lens fitting/reload/comparison were exercised on a tiny random Qwen fixture. This checks integration, not model quality.
- The full PyTorch train → held-out evaluation → base/adapted ordinary/J-lens readout workflow also completed on the real pretrained Qwen model.
- A real pretrained Qwen checkpoint also completed an exploratory two-prompt J-lens fit. This confirms execution and provenance, not readout faithfulness.
- CUDA/Windows training and large-model memory behavior remain unverified. Start with a small model and short sequences; there is no dependable peak-memory predictor yet.

Run the existing npm/Python/Rust checks. Build the desktop app with `npm run tauri build`. On this development machine use the isolated Rust environment shown in the local verification report.
