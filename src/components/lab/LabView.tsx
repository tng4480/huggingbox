import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { useAppStore } from "../../stores/appStore";
import { useWorkspaceDraft } from "../../hooks/useWorkspaceDraft";
import { listDownloadedModels } from "../../services/modelStorage";
import type { DownloadedModel } from "../../stores/appStore";
import Button from "../shared/Button";

type Run = {
  id: string;
  action: string;
  backend: string;
  task?: string;
  status: string;
  checkpointAvailable?: boolean;
  path: string;
};
type Cell = {
  layer: number;
  position: number;
  top: { token: string; score: number }[];
};
type Result = {
  counts?: Record<string, number>;
  preview?: unknown[];
  message?: string;
  adapter_path?: string;
  scores?: Record<string, number>;
  metric?: string;
  task?: string;
  classification_metrics?: Record<
    string,
    {
      accuracy: number;
      macro_f1: number;
      labels: string[];
      confusion_matrix: number[][];
    }
  >;
  rows?: {
    prompt: { role: string; content: string }[];
    expected: string;
    base: { text: string };
    adapter: { text: string };
  }[];
  tokens?: string[];
  views?: Record<string, Cell[]>;
  lens_path?: string;
  calibration_count?: number;
  meaning?: string;
};
type Detail = {
  result: Result | null;
  error: { message: string } | null;
  log: string;
  path: string;
  manifest?: unknown;
};
const runLabel = (r: Run) =>
  `${new Date(Number(r.id.split("-")[0])).toLocaleString()} · ${r.backend}`;
const sample = Array.from({ length: 12 }, (_, i) =>
  JSON.stringify({
    prompt: `Return the category for ${["apple", "pear", "peach", "plum", "orange", "melon", "car", "bus", "train", "boat", "bike", "truck"][i]}.`,
    completion: i < 6 ? "fruit" : "vehicle",
  }),
).join("\n");

export default function LabView() {
  const [task, setTask] = useWorkspaceDraft("lab", "task", "text-generation");
  const [trainingMethod, setTrainingMethod] = useWorkspaceDraft(
    "lab",
    "trainingMethod",
    "head",
  );
  const classification = task !== "text-generation";
  const [tab, setTab] = useWorkspaceDraft("lab", "tab", "train");
  const [model, setModel] = useWorkspaceDraft("lab", "model", "");
  const [backend, setBackend] = useWorkspaceDraft(
    "lab",
    "backend",
    navigator.platform.toLowerCase().includes("mac") ? "mlx" : "torch",
  );
  const [data, setData] = useWorkspaceDraft("lab", `dataset-${task}`, "");
  const [steps, setSteps] = useWorkspaceDraft("lab", "steps", 100);
  const [length, setLength] = useWorkspaceDraft("lab", "length", 256);
  const [rank, setRank] = useWorkspaceDraft("lab", "rank", 8);
  const [rate, setRate] = useWorkspaceDraft("lab", "rate", 0.0001);
  const [seed, setSeed] = useWorkspaceDraft("lab", "seed", 42);
  const [python, setPython] = useWorkspaceDraft("lab", "python", "");
  const [selected, setSelected] = useWorkspaceDraft("lab", "selected", "");
  const [training, setTraining] = useWorkspaceDraft("lab", "training", "");
  const [inspectionRun, setInspectionRun] = useWorkspaceDraft(
    "lab",
    "inspectionRun",
    "",
  );
  const [resume, setResume] = useWorkspaceDraft("lab", "resume", "");
  const [prompt, setPrompt] = useWorkspaceDraft(
    "lab",
    "prompt",
    "The currency of Japan is",
  );
  const [lens, setLens] = useWorkspaceDraft("lab", "lens", "");
  const [calibration, setCalibration] = useWorkspaceDraft(
    "lab",
    "calibration",
    "",
  );
  const [draftReady, setDraftReady] = useState(false);
  const [preparedModels, setPreparedModels] = useState<DownloadedModel[]>([]);
  const storagePath = useAppStore((s) => s.settings.modelStoragePath);
  const [runs, setRuns] = useState<Run[]>([]);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const [kind, setKind] = useState("ordinary");
  const [cell, setCell] = useState<Cell | null>(null);
  const inferenceJob = useAppStore((s) => s.activeJobId);
  useEffect(() => {
    let live = true;
    void listDownloadedModels(storagePath)
      .then((rows) => {
        if (live) setPreparedModels(rows);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [storagePath]);
  useEffect(() => {
    let live = true;
    void invoke<string>("lab_dataset_draft", { text: null, task })
      .then((value) => {
        if (live) {
          if (!useAppStore.getState().workspaceDrafts.lab?.[`dataset-${task}`])
            setData(value);
          setDraftReady(true);
        }
      })
      .catch((e) => {
        if (live) {
          setError(String(e));
          setDraftReady(true);
        }
      });
    return () => {
      live = false;
    };
  }, [task]);
  useEffect(() => {
    if (!draftReady) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      void invoke("lab_dataset_draft", { text: data, task }).catch((e) => {
        if (!cancelled) setError(String(e));
      });
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [data, draftReady, task]);
  const active = runs.find((r) => r.status === "running");
  const busy = starting || Boolean(active) || Boolean(inferenceJob);
  const resumable = runs.filter(
    (r) =>
      r.action === "train" &&
      r.status !== "running" &&
      r.checkpointAvailable &&
      (r.task || "text-generation") === task,
  );
  const trained = runs.filter(
    (r) =>
      r.action === "train" &&
      r.status === "completed" &&
      (r.task || "text-generation") === task,
  );
  useEffect(() => {
    let live = true;
    let pending = false;
    async function refresh() {
      if (pending) return;
      pending = true;
      try {
        const rows = await invoke<Run[]>("lab_runs");
        if (!live) return;
        setRuns(rows);
        if (selected) {
          const d = await invoke<Detail>("lab_run_detail", { id: selected });
          if (live) setDetail(d);
        }
      } catch (e) {
        if (live) setError(String(e));
      } finally {
        pending = false;
      }
    }
    void refresh();
    const timer = setInterval(refresh, 1500);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [selected]);
  async function start(action: string) {
    setError("");
    setStarting(true);
    try {
      const id = await invoke<string>("lab_start", {
        config: {
          action,
          task,
          training_method: trainingMethod,
          backend:
            action === "evaluate"
              ? trained.find((r) => r.id === (training || trained[0]?.id))
                  ?.backend || backend
              : classification
                ? "torch"
                : backend,
          model_path: model,
          dataset: data,
          steps,
          max_length: length,
          rank,
          layers: 4,
          learning_rate: rate,
          seed,
          max_tokens: 64,
          python_path: python,
          run_id: training || trained[0]?.id || null,
          resume_id: classification ? null : resume || null,
          inspection_run_id: inspectionRun || null,
          prompt,
          lens_path: lens || null,
          calibration,
        },
      });
      setSelected(id);
      setDetail(null);
      setCell(null);
      setRuns(await invoke<Run[]>("lab_runs"));
    } catch (e) {
      setError(String(e));
    } finally {
      setStarting(false);
    }
  }
  async function choose(
    kind: "model" | "dataset" | "lens" | "python" | "images",
  ) {
    try {
      const path = await open({
        directory: kind === "model" || kind === "images",
        multiple: false,
        title: `Choose ${kind}`,
      });
      if (typeof path !== "string") return;
      if (kind === "images")
        setData(await invoke<string>("lab_import_image_folder", { path }));
      else if (kind === "model") setModel(path);
      else if (kind === "lens") setLens(path);
      else if (kind === "python") setPython(path);
      else setData(await invoke<string>("lab_read_dataset", { path }));
    } catch (e) {
      setError(String(e));
    }
  }
  const result = detail?.result;
  const selectedRun = runs.find((r) => r.id === selected);
  const metrics = (detail?.log || "")
    .split("\n")
    .filter((l) => l.startsWith("HB_LAB:"))
    .flatMap((l) => {
      try {
        return [JSON.parse(l.slice(7))];
      } catch {
        return [];
      }
    });
  const loss = metrics.filter(
    (m) => m.kind === "metric" && Number.isFinite(m.train_loss),
  );
  const maxLoss = Math.max(1, ...loss.map((m) => m.train_loss));
  const viewKind = result?.views?.[kind]
    ? kind
    : Object.keys(result?.views || {})[0] || kind;
  const cells = result?.views?.[viewKind] || [];
  const layers = [...new Set(cells.map((c) => c.layer))];
  return (
    <div className="lab-view">
      <header className="lab-title">
        <div>
          <span className="lab-eyebrow">MODEL LAB</span>
          <h1>Improve. Compare. Inspect.</h1>
          <p>
            Train task-specific models, compare held-out examples, then inspect
            supported language-model readouts.
          </p>
        </div>
        <span className="lab-badge">Local · original weights preserved</span>
      </header>
      <nav className="lab-tabs" aria-label="Lab workflow">
        {[
          ["train", "1 · Fine-tune"],
          ["compare", "2 · Compare"],
          ["inspect", "3 · Inspect"],
        ].map(([value, label]) => (
          <button
            key={value}
            disabled={value === "inspect" && classification}
            title={
              value === "inspect" && classification
                ? "Inspection supports language decoders only"
                : undefined
            }
            aria-current={tab === value ? "page" : undefined}
            onClick={() => {
              setTab(value);
              setCell(null);
            }}
          >
            {label}
          </button>
        ))}
      </nav>
      {error && (
        <div role="alert" className="lab-error">
          {error}
        </div>
      )}
      <div className="lab-layout">
        <section className="lab-card">
          <label>
            Training task
            <select
              value={task}
              disabled={busy}
              onChange={(e) => {
                setTask(e.target.value);
                setSelected("");
                setDetail(null);
                setError("");
                setTraining("");
                setResume("");
                setTab("train");
                setDraftReady(false);
                if (e.target.value !== "text-generation") setBackend("torch");
              }}
            >
              <option value="text-generation">Text generation · LoRA</option>
              <option value="text-classification">Text classification</option>
              <option value="image-classification">Image classification</option>
              <option disabled>Speech recognition · not yet supported</option>
              <option disabled>Embeddings · not yet supported</option>
              <option disabled>Diffusion training · not yet supported</option>
            </select>
          </label>
          <p className="lab-muted">
            Train: selected task, compatible checkpoints only. Inspect: language
            decoders only. Inference support does not guarantee training
            support.
          </p>
          <label>
            Prepared model
            <select
              value={
                preparedModels.find((m) => model === m.storagePath)
                  ?.storagePath || ""
              }
              onChange={(e) => {
                if (e.target.value) setModel(e.target.value);
              }}
              disabled={busy}
            >
              <option value="">
                Choose a downloaded model or browse a local folder
              </option>
              {preparedModels.map((m) => (
                <option key={m.id} value={m.storagePath}>
                  {m.id}
                </option>
              ))}
            </select>
          </label>
          <label>
            Local model folder
            <div className="lab-input-row">
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="Choose prepared Safetensors or MLX model"
                disabled={busy}
              />
              <Button onClick={() => void choose("model")} disabled={busy}>
                Browse
              </Button>
            </div>
          </label>
          <label>
            Runtime
            <select
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
              disabled={busy}
            >
              <option value="mlx" disabled={classification}>
                MLX · Apple Silicon training
              </option>
              <option value="torch">
                PyTorch · CPU / NVIDIA · training & inspection
              </option>
            </select>
          </label>
          <p className="lab-muted">
            MLX quantized models support adapter training. PyTorch starts with
            full-precision weights; GGUF and arbitrary architectures are not
            supported. Inspection needs a Transformers checkpoint.
          </p>
          <Button
            onClick={() => void start("setup")}
            disabled={busy || Boolean(python)}
          >
            Prepare isolated runtime
          </Button>
          {tab === "train" && (
            <>
              <h2>Training examples</h2>
              <p>
                {classification
                  ? task === "text-classification"
                    ? 'JSONL: {"text":"example","label":"class"}. At least six distinct examples per class.'
                    : 'Import class folders or JSONL: {"image":"/absolute/path.png","label":"class"}. At least six images per class.'
                  : "JSONL: prompt/completion pairs or chat messages ending in an assistant answer. At least six distinct inputs."}
              </p>
              <div className="lab-actions">
                {task === "image-classification" && (
                  <Button onClick={() => void choose("images")} disabled={busy}>
                    Import class folders
                  </Button>
                )}
                <Button onClick={() => void choose("dataset")} disabled={busy}>
                  Import JSONL
                </Button>
                <Button
                  onClick={() =>
                    setData(
                      classification
                        ? Array.from({ length: 12 }, (_, i) =>
                            JSON.stringify({
                              text: `${i < 6 ? "Positive" : "Negative"} review example ${i}`,
                              label: i < 6 ? "positive" : "negative",
                            }),
                          ).join("\n")
                        : sample,
                    )
                  }
                  disabled={
                    busy || Boolean(data) || task === "image-classification"
                  }
                >
                  Load toy example
                </Button>
              </div>
              <textarea
                aria-label="Training dataset"
                className="lab-dataset"
                value={data}
                onChange={(e) => setData(e.target.value)}
                disabled={busy || !draftReady}
                placeholder={
                  task === "image-classification"
                    ? '{"image":"/absolute/path.png","label":"class"}'
                    : task === "text-classification"
                      ? '{"text":"Example","label":"class"}'
                      : '{"prompt":"Your question","completion":"Desired answer"}'
                }
              />
              <div className="lab-fields">
                <label>
                  Training steps
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    disabled={busy}
                  />
                </label>
                {task !== "image-classification" && (
                  <label>
                    Maximum tokens/example
                    <input
                      type="number"
                      min={32}
                      max={2048}
                      value={length}
                      onChange={(e) => setLength(Number(e.target.value))}
                      disabled={busy}
                    />
                  </label>
                )}
              </div>
              <details>
                <summary>Training settings</summary>
                {classification && (
                  <label>
                    Training method
                    <select
                      value={trainingMethod}
                      onChange={(e) => setTrainingMethod(e.target.value)}
                    >
                      <option value="head">
                        Train classification head only
                      </option>
                      <option value="full">Fine-tune full model</option>
                    </select>
                  </label>
                )}
                <div className="lab-fields">
                  {!classification && (
                    <label>
                      Adapter rank (language models only)
                      <input
                        type="number"
                        min={1}
                        max={64}
                        value={rank}
                        onChange={(e) => setRank(Number(e.target.value))}
                      />
                    </label>
                  )}
                  <label>
                    Learning rate
                    <input
                      type="number"
                      step={0.00001}
                      value={rate}
                      onChange={(e) => setRate(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Split seed
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                    />
                  </label>
                </div>
                {!classification && (
                  <label>
                    Continue from an adapter checkpoint
                    <select
                      value={resume}
                      onChange={(e) => setResume(e.target.value)}
                    >
                      <option value="">Start fresh</option>
                      {resumable
                        .filter((r) => r.backend === backend)
                        .map((r) => (
                          <option key={r.id} value={r.id}>
                            {runLabel(r)}
                          </option>
                        ))}
                    </select>
                  </label>
                )}
                {!classification && (
                  <p className="lab-muted">
                    Continuation uses the latest readable checkpoint with a
                    fresh optimizer. Keep the same dataset and seed.
                  </p>
                )}
              </details>
              <div className="lab-actions">
                <Button
                  onClick={() => void start("validate")}
                  disabled={busy || !data}
                >
                  Validate & preview
                </Button>
                <Button
                  variant="primary"
                  onClick={() => void start("train")}
                  disabled={busy || !model || !data}
                >
                  {classification ? "Train classifier" : "Train adapter"}
                </Button>
              </div>
              <p className="lab-muted">
                Fixed split: about 80% training, 10% validation, 10% test.
                Duplicate inputs are rejected. Classification splits preserve
                every class. Text examples are never silently truncated.
              </p>
            </>
          )}
          {tab === "compare" && (
            <>
              <h2>
                {classification
                  ? "Before vs after training"
                  : "Original vs adapter"}
              </h2>
              <p>
                Both models get the same untouched test examples. Test data
                never enters training or validation.
              </p>
              <label>
                Completed training run
                <select
                  value={training || trained[0]?.id || ""}
                  onChange={(e) => setTraining(e.target.value)}
                >
                  <option value="" disabled>
                    Select a trained adapter
                  </option>
                  {trained.map((r) => (
                    <option key={r.id} value={r.id}>
                      {runLabel(r)}
                    </option>
                  ))}
                </select>
              </label>
              <Button
                variant="primary"
                onClick={() => void start("evaluate")}
                disabled={busy || !trained.length}
              >
                Evaluate held-out examples
              </Button>
              <p className="lab-muted">
                {classification
                  ? "Compare accuracy, macro F1 and per-class errors. Baseline uses the same newly initialised head before training. Classification exports include the complete model, labels and preprocessing."
                  : "The run’s recorded model, adapter and backend are used. Exact match is useful for constrained tasks; read the examples to judge broader quality."}
              </p>
            </>
          )}
          {tab === "inspect" && (
            <>
              <h2>Inside the model</h2>
              {classification && (
                <p role="status">
                  Inspection is available only for supported language decoders.
                  Classification models can be trained and evaluated here.
                </p>
              )}
              <p>
                Read vocabulary associations at selected layers and token
                positions. These are not thoughts, confidence estimates or proof
                of causation.
              </p>
              <label>
                Compare adapter readouts (optional)
                <select
                  value={inspectionRun}
                  onChange={(e) => setInspectionRun(e.target.value)}
                >
                  <option value="">Original model only</option>
                  {trained
                    .filter((r) => r.backend === "torch")
                    .map((r) => (
                      <option key={r.id} value={r.id}>
                        {runLabel(r)}
                      </option>
                    ))}
                </select>
              </label>
              <label>
                Prompt
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  disabled={busy}
                />
              </label>
              <label>
                Matching J-lens file (optional)
                <div className="lab-input-row">
                  <input
                    value={lens}
                    onChange={(e) => setLens(e.target.value)}
                    placeholder="lens.pt with its HuggingBox manifest"
                  />
                  <Button onClick={() => void choose("lens")} disabled={busy}>
                    Browse
                  </Button>
                </div>
              </label>
              <Button
                variant="primary"
                onClick={() => void start("inspect")}
                disabled={
                  busy || !model || backend !== "torch" || classification
                }
              >
                Inspect prompt
              </Button>
              <details>
                <summary>Fit an exploratory J-lens</summary>
                <p>
                  Fit two sampled layers to this exact checkpoint. This needs
                  backward passes, can be slow, and is not a validated
                  explanation. Use independent prompts to check the result.
                </p>
                <label>
                  Calibration prompts · one per line
                  <textarea
                    value={calibration}
                    onChange={(e) => setCalibration(e.target.value)}
                    placeholder="2–100 short, varied prompts"
                  />
                </label>
                <Button
                  onClick={() => void start("fit_lens")}
                  disabled={
                    busy || !model || backend !== "torch" || !calibration
                  }
                >
                  Fit lens & inspect
                </Button>
              </details>
            </>
          )}
          <details>
            <summary>Advanced runtime</summary>
            <label>
              Existing Python executable
              <div className="lab-input-row">
                <input
                  value={python}
                  onChange={(e) => setPython(e.target.value)}
                  placeholder="Optional; must already have the selected runtime"
                />
                <Button onClick={() => void choose("python")}>Browse</Button>
              </div>
            </label>
            <p className="lab-muted">
              Leave blank to use the isolated Lab environment. Setup never
              modifies an overridden environment.
            </p>
          </details>
        </section>
        <section className="lab-results">
          <div className="lab-card">
            <div className="lab-section-title">
              <h2>Runs & results</h2>
              {(starting || active) && (
                <Button
                  variant="danger"
                  onClick={() =>
                    void invoke("lab_cancel").catch((e) => setError(String(e)))
                  }
                >
                  Stop job
                </Button>
              )}
            </div>
            <label>
              Saved run
              <select
                value={selected}
                onChange={(e) => {
                  setSelected(e.target.value);
                  setCell(null);
                }}
              >
                <option value="">Choose a run</option>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.action} · {r.status} · {runLabel(r)}
                  </option>
                ))}
              </select>
            </label>
            {selectedRun && (
              <p role="status">
                <span className="lab-badge">{selectedRun.status}</span>{" "}
                {selectedRun.action} · {selectedRun.backend}
              </p>
            )}
            {!selected && (
              <div className="lab-empty">
                <h3>Your experiments stay here.</h3>
                <p>
                  Start with a small dataset and a short training run. A
                  successful training job is only the first step—compare
                  held-out results before using the adapter.
                </p>
              </div>
            )}
            {detail?.error && (
              <p className="lab-error" role="alert">
                {detail.error.message}
              </p>
            )}
            {result?.counts && (
              <>
                <div className="lab-score-grid">
                  {Object.entries(result.counts).map(([k, v]) => (
                    <div key={k}>
                      <strong>{v}</strong>
                      <span>{k} examples</span>
                    </div>
                  ))}
                </div>
                <h3>Example preview</h3>
                <pre>{JSON.stringify(result.preview, null, 2)}</pre>
              </>
            )}
            {result?.message && <p>{result.message}</p>}
            {loss.length > 0 && (
              <>
                <h3>Training loss</h3>
                <svg
                  viewBox="0 0 500 130"
                  role="img"
                  aria-label={`Training loss: ${loss[0].train_loss.toFixed(3)} to ${loss[loss.length - 1].train_loss.toFixed(3)}`}
                >
                  <polyline
                    fill="none"
                    stroke="var(--accent-primary)"
                    strokeWidth="2"
                    points={loss
                      .map(
                        (m, i) =>
                          `${10 + (i * 480) / Math.max(1, loss.length - 1)},${120 - (m.train_loss / maxLoss) * 110}`,
                      )
                      .join(" ")}
                  />
                </svg>
                <p className="lab-muted">
                  Latest loss {loss[loss.length - 1].train_loss.toFixed(3)} ·
                  lower training loss does not prove better held-out
                  performance.
                </p>
              </>
            )}
            {result?.scores && (
              <>
                <div className="lab-score-grid">
                  {["base", "adapter"]
                    .map(
                      (name) =>
                        [name, result.scores![name]] as [string, number],
                    )
                    .map(([name, score]) => (
                      <div key={name}>
                        <strong>{(score * 100).toFixed(1)}%</strong>
                        <span>
                          {result.classification_metrics
                            ? (name === "base"
                                ? "Before training"
                                : "Trained model") + " accuracy"
                            : name + " exact match"}
                        </span>
                      </div>
                    ))}
                </div>
                <p>{result.metric}</p>
                {result.classification_metrics &&
                  Object.entries(result.classification_metrics).map(
                    ([name, m]) => (
                      <section key={name}>
                        <h3>
                          {name === "base"
                            ? "Before training"
                            : "Trained model"}{" "}
                          · Macro F1 {(m.macro_f1 * 100).toFixed(1)}%
                        </h3>
                        <div className="lens-grid">
                          <table>
                            <caption>
                              Confusion matrix: rows = actual; columns =
                              predicted
                            </caption>
                            <thead>
                              <tr>
                                <th>Actual / predicted</th>
                                {m.labels.map((l) => (
                                  <th key={l}>{l}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {m.confusion_matrix.map((row, i) => (
                                <tr key={i}>
                                  <th>{m.labels[i]}</th>
                                  {row.map((v, j) => (
                                    <td key={j}>{v}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>
                    ),
                  )}
                {result.rows?.map((row, i) => (
                  <article key={i} className="lab-example">
                    <h3>Held-out example {i + 1}</h3>
                    <p>
                      {row.prompt
                        .map((m) => `${m.role}: ${m.content}`)
                        .join("\n")}
                    </p>
                    <p>
                      <b>Expected:</b> {row.expected}
                    </p>
                    <div className="lab-fields">
                      <div>
                        <b>Original</b>
                        <pre>{row.base.text}</pre>
                      </div>
                      <div>
                        <b>
                          {result.classification_metrics
                            ? "Trained model"
                            : "Adapter"}
                        </b>
                        <pre>{row.adapter.text}</pre>
                      </div>
                    </div>
                  </article>
                ))}
              </>
            )}
            {result?.views && (
              <>
                {result.calibration_count != null && (
                  <p className="lab-badge">
                    Exploratory lens · {result.calibration_count} calibration
                    prompts · sampled layers
                  </p>
                )}
                <p>{result.meaning}</p>
                <label>
                  Readout
                  <select
                    value={viewKind}
                    onChange={(e) => {
                      setKind(e.target.value);
                      setCell(null);
                    }}
                  >
                    {Object.keys(result.views).map((k) => (
                      <option key={k}>{k}</option>
                    ))}
                  </select>
                </label>
                <div className="lens-grid">
                  <table>
                    <caption>
                      Click a layer/token cell to inspect its top five
                      vocabulary scores.
                    </caption>
                    <thead>
                      <tr>
                        <th>Layer</th>
                        {result.tokens?.map((t, i) => (
                          <th key={i}>
                            {i}: {t}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {layers.map((layer) => (
                        <tr key={layer}>
                          <th>{layer}</th>
                          {result.tokens?.map((_, position) => {
                            const c = cells.find(
                              (v) =>
                                v.layer === layer && v.position === position,
                            );
                            return (
                              <td key={position}>
                                <button
                                  className={cell === c ? "selected" : ""}
                                  onClick={() => setCell(c || null)}
                                  aria-label={`Layer ${layer}, token ${position}: ${c?.top[0].token}`}
                                >
                                  {c?.top[0].token || "—"}
                                </button>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {cell && (
                  <ol>
                    {cell.top.map((v, i) => (
                      <li key={i}>
                        {JSON.stringify(v.token)}{" "}
                        <span className="lab-muted">
                          score {v.score.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
                {result.lens_path && (
                  <Button
                    onClick={() => {
                      setLens(result.lens_path!);
                      setTab("inspect");
                    }}
                  >
                    Use this fitted lens
                  </Button>
                )}
              </>
            )}
            {detail && (
              <>
                <details>
                  <summary>Progress & diagnostics</summary>
                  <pre className="lab-log">{detail.log || "Starting…"}</pre>
                </details>
                <Button
                  onClick={() =>
                    void revealItemInDir(detail.path).catch((e) =>
                      setError(String(e)),
                    )
                  }
                >
                  Show run folder / adapter files
                </Button>
              </>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
