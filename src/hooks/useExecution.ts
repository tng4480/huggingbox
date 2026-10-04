import { useEffect, useRef, useCallback } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../stores/appStore';
import { listDownloadedModels } from '../services/modelStorage';
import { listModelEnvironments } from '../services/modelEnvironments';
import { confirmDialog, messageDialog } from '../services/dialogs';
import {
  clearPreferredEnvModelId,
  getPreferredEnvModelId,
  setPreferredEnvModelId,
} from '../services/modelEnvPreference';

interface ModelDependencyProbeResult {
  missingPackages: string[];
  requiredPackages?: string[];
  compatibilityError?: string | null;
}

interface ExecutionStatsPayload {
  pid: number;
  rss_bytes: number;
  virtual_memory_bytes: number;
  gpu_memory_bytes?: number | null;
  elapsed_ms: number;
}

let executionListenerUsers = 0;
let executionListenersCleanup: (() => void) | null = null;
let executionListenersInitPromise: Promise<void> | null = null;
let executionTimer: ReturnType<typeof setInterval> | null = null;
let bufferedStdout = '';
let bufferedStderr = '';
let bufferedStreamFlushTimer: ReturnType<typeof setTimeout> | null = null;

function timestamp(): string {
  return new Date().toLocaleTimeString();
}

function appendLog(text: string): void {
  useAppStore
    .getState()
    .appendExecutionOutput(`[HuggingBox ${timestamp()}] ${text.endsWith('\n') ? text : `${text}\n`}`);
}

function flushBufferedStreams(): void {
  if (bufferedStreamFlushTimer) {
    clearTimeout(bufferedStreamFlushTimer);
    bufferedStreamFlushTimer = null;
  }

  const store = useAppStore.getState();
  if (bufferedStdout) {
    store.appendExecutionOutput(bufferedStdout);
    bufferedStdout = '';
  }
  if (bufferedStderr) {
    store.appendStderrOutput(bufferedStderr);
    bufferedStderr = '';
  }
}

function resetBufferedStreams(): void {
  if (bufferedStreamFlushTimer) {
    clearTimeout(bufferedStreamFlushTimer);
    bufferedStreamFlushTimer = null;
  }
  bufferedStdout = '';
  bufferedStderr = '';
}

function scheduleBufferedStreamFlush(): void {
  if (bufferedStreamFlushTimer) return;
  bufferedStreamFlushTimer = setTimeout(() => {
    bufferedStreamFlushTimer = null;
    flushBufferedStreams();
  }, 75);
}

function queueStdout(text: string): void {
  applyMetricMarkers(text);
  bufferedStdout += text;
  scheduleBufferedStreamFlush();
}

function queueStderr(text: string): void {
  bufferedStderr += text;
  scheduleBufferedStreamFlush();
}

function applyMetricMarkers(text: string): void {
  const store = useAppStore.getState();
  const runtimeMatches = Array.from(text.matchAll(/HB_RUNTIME:(.+)/g));
  const providerMatches = Array.from(text.matchAll(/HB_PROVIDER:(.+)/g));
  const inferenceMatches = Array.from(text.matchAll(/HB_METRIC_INFERENCE_SECONDS:(.+)/g));
  const tokenMatches = Array.from(text.matchAll(/HB_METRIC_TOKENS:(.+)/g));

  const patch: Parameters<typeof store.setExecutionStats>[0] = {};

  const runtime = runtimeMatches[runtimeMatches.length - 1]?.[1]?.trim();
  if (runtime) patch.executionRuntime = runtime;

  const provider = providerMatches[providerMatches.length - 1]?.[1]?.trim();
  if (provider) patch.executionProvider = provider;

  const inferenceText = inferenceMatches[inferenceMatches.length - 1]?.[1]?.trim();
  if (inferenceText) {
    const inferenceSeconds = Number(inferenceText);
    if (Number.isFinite(inferenceSeconds) && inferenceSeconds >= 0) {
      patch.inferenceSeconds = inferenceSeconds;
    }
  }

  const tokenText = tokenMatches[tokenMatches.length - 1]?.[1]?.trim();
  if (tokenText) {
    const tokenCount = Number(tokenText);
    const inferenceSeconds =
      patch.inferenceSeconds ??
      useAppStore.getState().executionStats.inferenceSeconds;
    if (Number.isFinite(tokenCount) && tokenCount >= 0 && inferenceSeconds && inferenceSeconds > 0) {
      patch.tokensPerSecond = tokenCount / inferenceSeconds;
    }
  }

  if (Object.keys(patch).length > 0) {
    store.setExecutionStats(patch);
  }
}

function normalizeDependencyName(raw: string): string {
  const token = raw.trim().replace(/^['"`]+|['"`]+$/g, '');
  const lower = token.toLowerCase();
  if (!lower) return '';
  if (lower === 'pil') return 'pillow';
  if (lower === 'sklearn') return 'scikit-learn';
  if (lower === 'cv2') return 'opencv-python';
  return lower;
}

function uniqueDependencies(items: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const normalized = normalizeDependencyName(item);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function requirementPackageName(requirement: string): string {
  const trimmed = requirement.trim();
  const match = trimmed.match(/^[A-Za-z][A-Za-z0-9._-]*/);
  return normalizeDependencyName(match?.[0] ?? trimmed);
}

function uniqueRequirementSpecs(items: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const raw = item.trim();
    const key = requirementPackageName(raw);
    if (!raw || !key || seen.has(key)) continue;
    seen.add(key);
    out.push(raw);
  }
  return out;
}

function filterRequirementSpecsByMissing(specs: string[], missingPackages: string[]): string[] {
  const missing = new Set(missingPackages.map((item) => normalizeDependencyName(item)));
  return specs.filter((spec) => missing.has(requirementPackageName(spec)));
}

function hasVersionSpecifier(requirement: string): boolean {
  return /[<>=!~]/.test(requirement);
}

function detectRuntimeTypeFromCode(code: string | null | undefined): string {
  const raw = (code ?? '').trim();
  if (!raw) return 'transformers_llm';
  const match = raw.match(/^\s*#?\s*RUNTIME:\s*([A-Za-z0-9._-]+)/im);
  if (!match?.[1]) return 'transformers_llm';
  return match[1].trim().toLowerCase();
}

function runtimeDependenciesForType(runtimeType: string): string[] {
  const runtime = runtimeType.trim().toLowerCase();
  if (!runtime) return ['transformers'];
  if (runtime === 'transformers_audio') {
    return ['transformers', 'accelerate', 'torch', 'librosa', 'soundfile', 'sentencepiece'];
  }
  if (runtime === 'transformers_multimodal') {
    return ['transformers', 'accelerate', 'torch', 'torchvision', 'pillow', 'sentencepiece'];
  }
  if (runtime.startsWith('transformers')) {
    return ['transformers'];
  }
  if (runtime.startsWith('diffusers')) {
    return ['diffusers', 'transformers', 'accelerate', 'torch', 'torchvision', 'pillow'];
  }
  if (runtime === 'onnxruntime') {
    return ['optimum', 'onnxruntime', 'transformers', 'torch', 'pillow'];
  }
  if (runtime.startsWith('sentence_transformers')) {
    return ['sentence-transformers', 'transformers'];
  }
  return ['transformers'];
}

function modelRunOnceKey(modelId: string): string {
  return `huggingbox:model-run-once:${modelId}`;
}

function hasModelRunOnce(modelId: string): boolean {
  try {
    return localStorage.getItem(modelRunOnceKey(modelId)) === '1';
  } catch {
    return false;
  }
}

function markModelRunOnce(modelId: string): void {
  try {
    localStorage.setItem(modelRunOnceKey(modelId), '1');
  } catch {
    // ignore storage failures
  }
}

async function chooseExecutionEnvModelId(
  modelId: string,
  envStoragePath?: string
): Promise<string | null> {
  const envs = await listModelEnvironments(envStoragePath);
  const envIds = envs.map((e) => e.modelId);
  const hasOwnEnv = envIds.includes(modelId);

  if (hasOwnEnv) {
    setPreferredEnvModelId(modelId, modelId);
    appendLog(`Using model environment: ${modelId}`);
    return modelId;
  }

  const preferred = getPreferredEnvModelId(modelId);
  if (preferred && envIds.includes(preferred)) {
    appendLog(`Using saved shared environment: ${preferred}`);
    return preferred;
  }
  if (preferred && !envIds.includes(preferred)) {
    clearPreferredEnvModelId(modelId);
    appendLog(`Saved shared environment not found anymore: ${preferred}`);
  }

  const createNew = await confirmDialog(
    `No Python environment exists for this model yet:\n${modelId}\n\nCreate a new isolated environment? (Recommended)\n\nClick Cancel to use an existing environment.`
  );
  if (createNew) {
    setPreferredEnvModelId(modelId, modelId);
    appendLog('Environment choice: create new isolated environment.');
    return modelId;
  }

  const candidates = envIds.filter((id) => id !== modelId);
  if (candidates.length === 0) {
    await messageDialog('No existing environments were found. Creating a new isolated environment instead.', {
      kind: 'info',
    });
    setPreferredEnvModelId(modelId, modelId);
    appendLog('No reusable environments found; falling back to new isolated environment.');
    return modelId;
  }

  const preview = candidates.slice(0, 12).join('\n');
  const picked = window.prompt(
    `Choose an existing environment by model id:\n\n${preview}\n\nEnter exact model id:`,
    candidates[0]
  );
  if (picked === null) {
    appendLog('Environment selection cancelled by user.');
    return null;
  }

  const chosen = picked.trim();
  if (!chosen) {
    appendLog('Environment selection cancelled (empty input).');
    return null;
  }
  if (!candidates.includes(chosen)) {
    await messageDialog(`Environment "${chosen}" was not found in the existing environment list.`, {
      kind: 'warning',
    });
    appendLog(`Invalid shared environment selected: ${chosen}`);
    return null;
  }

  setPreferredEnvModelId(modelId, chosen);
  appendLog(`Environment choice: reuse existing environment ${chosen}`);
  return chosen;
}

async function ensureExecutionListenersRegistered(): Promise<void> {
  if (executionListenersCleanup) return;
  if (executionListenersInitPromise) {
    await executionListenersInitPromise;
    return;
  }

  executionListenersInitPromise = (async () => {
    const unlistens: Array<() => void> = [];

    const u1 = await listen<{ text: string }>('execution-stdout', (e) => {
      queueStdout(e.payload.text);
    });

    const u2 = await listen<{ text: string }>('execution-stderr', (e) => {
      queueStderr(e.payload.text);
    });

    const u3 = await listen<{ exit_code: number }>('execution-done', (e) => {
      flushBufferedStreams();
      const state =
        e.payload.exit_code === 0
          ? 'completed'
          : e.payload.exit_code === -2
            ? 'cancelled'
            : 'error';
      if (executionTimer) {
        clearInterval(executionTimer);
        executionTimer = null;
      }
      useAppStore.getState().setExecutionState(state);
      appendLog(`Execution finished with exit code ${e.payload.exit_code}.`);
    });

    const u4 = await listen<{ text: string }>('install-progress', (e) => {
      queueStdout(e.payload.text);
    });

    const u5 = await listen<{ text: string }>('download-progress', (e) => {
      queueStdout(e.payload.text);
    });

    const u6 = await listen<{
      percent: number;
      downloaded_bytes: number;
      total_bytes: number;
      speed_bps: number;
      eta_seconds: number | null;
      phase: string;
      files_done: number;
      files_total: number;
      filename?: string;
    }>('download-stats', (e) => {
      useAppStore.getState().setDownloadStats({
        percent: e.payload.percent,
        downloadedBytes: e.payload.downloaded_bytes,
        totalBytes: e.payload.total_bytes,
        speedBps: e.payload.speed_bps,
        etaSeconds: e.payload.eta_seconds,
        phase: e.payload.phase,
        filesDone: e.payload.files_done,
        filesTotal: e.payload.files_total,
        filename: e.payload.filename,
      });
    });

    const u7 = await listen<ExecutionStatsPayload>('execution-stats', (e) => {
      useAppStore.getState().setExecutionStats({
        processRssBytes: e.payload.rss_bytes,
        processVirtualMemoryBytes: e.payload.virtual_memory_bytes,
        processGpuBytes: e.payload.gpu_memory_bytes ?? null,
      });
      useAppStore.getState().setExecutionElapsed(e.payload.elapsed_ms);
    });

    unlistens.push(u1, u2, u3, u4, u5, u6, u7);
    executionListenersCleanup = () => {
      for (const fn of unlistens) fn();
      executionListenersCleanup = null;
      executionListenersInitPromise = null;
    };
  })();

  await executionListenersInitPromise;
}

/**
 * Manages Python code execution via Tauri commands.
 * - Registers global listeners once
 * - Provides runCode() which can install dependencies for first-run only
 * - Provides cancelExecution()
 */
export function useExecution() {
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    executionListenerUsers += 1;

    void ensureExecutionListenersRegistered().catch(() => {
      if (!cancelled) {
        appendLog('Warning: failed to register execution listeners.');
      }
    });

    return () => {
      cancelled = true;
      executionListenerUsers = Math.max(0, executionListenerUsers - 1);
      if (executionListenerUsers === 0 && executionListenersCleanup) {
        executionListenersCleanup();
      }
    };
  }, []);

  function startTimer() {
    if (executionTimer) {
      timerRef.current = executionTimer;
      return;
    }
    const start = Date.now();
    useAppStore.getState().setExecutionStartTime(start);
    executionTimer = setInterval(() => {
      useAppStore.getState().setExecutionElapsed(Date.now() - start);
    }, 1000);
    timerRef.current = executionTimer;
  }

  function stopTimer() {
    if (executionTimer) {
      clearInterval(executionTimer);
      executionTimer = null;
    }
    if (timerRef.current) {
      timerRef.current = null;
    }
  }

  const runCode = useCallback(async (
    options?: {
      modelId?: string;
      storagePath?: string;
      hfToken?: string;
      pipelineTag?: string | null;
      preferredDevice?: 'auto' | 'cpu' | 'cuda';
      selectedGpuId?: string | null;
      userInput?: string;
      envStoragePath?: string;
      scriptRelativePath?: string;
      runtimeSourceCode?: string;
      runMode?: 'prepared' | 'direct';
      diffusionMode?: 'text-to-image' | 'image-to-image' | 'inpainting';
      outputDir?: string;
      negativePrompt?: string;
      steps?: number;
      guidanceScale?: number;
      seed?: string;
      numImages?: number;
      strength?: number;
      sourceImagePath?: string;
      maskImagePath?: string;
      multimodalTask?: 'visual-question-answering' | 'image-captioning' | 'document-understanding';
      imagePath?: string;
      documentPath?: string;
    }
  ) => {
    const store = useAppStore.getState();
    resetBufferedStreams();
    store.clearExecutionOutput();
    store.clearStderrOutput();
    store.clearExecutionStats();
    store.setExecutionError(null);
    store.setDownloadStats(null);
    store.setExecutionState('running');
    startTimer();

    const modelId = options?.modelId?.trim() || '';
    let executionEnvModelId: string | null = modelId || null;
    if (modelId) {
      store.setActiveExecutionModelId(modelId);

      try {
        executionEnvModelId = await chooseExecutionEnvModelId(modelId, options?.envStoragePath);
      } catch (error) {
        appendLog(`Environment discovery failed; defaulting to a new isolated environment. ${String(error)}`);
        executionEnvModelId = modelId;
      }

      if (!executionEnvModelId) {
        stopTimer();
        store.setExecutionState('idle');
        store.setExecutionError('Execution cancelled during Python environment selection.');
        store.setDownloadStats(null);
        return;
      }

      store.setActiveExecutionEnvModelId(executionEnvModelId);
    }

    const hasRunBefore = modelId ? hasModelRunOnce(modelId) : false;
    const allowAutoInstall = !hasRunBefore;
    if (modelId) {
      markModelRunOnce(modelId);
    }

    appendLog(`Run requested${modelId ? ` for ${modelId}` : ''}.`);
    appendLog(
      options?.runMode === 'direct'
        ? 'Run mode: direct execution (skipping app-managed dependency and compatibility checks).'
        : 'Run mode: prepared execution (dependency checks and compatibility probe enabled).'
    );
    appendLog(
      allowAutoInstall
        ? 'Auto dependency installation is enabled for this first run.'
        : 'Auto dependency installation is disabled for this model (manual environment mode).'
    );

    if (options?.runMode !== 'direct' && options?.modelId && options?.storagePath) {
      try {
        appendLog(`Checking download readiness in ${options.storagePath}`);
        const missingDownloadDeps = await invoke<string[]>('check_packages', {
          packages: ['huggingface_hub'],
          modelId: options.modelId,
          venvModelId: executionEnvModelId,
        });

        if (missingDownloadDeps.length > 0 && !allowAutoInstall) {
          stopTimer();
          store.setExecutionState('error');
          store.setExecutionError(
            `Download dependencies missing (${missingDownloadDeps.join(', ')}). Auto installs are disabled after first run for this model. Install manually from the terminal panel.`
          );
          store.setDownloadStats(null);
          appendLog(`Missing download dependencies with auto-install disabled: ${missingDownloadDeps.join(', ')}`);
          return;
        }

        if (missingDownloadDeps.length > 0) {
          const approveDeps = await confirmDialog(
            `Model downloads require ${missingDownloadDeps.join(', ')}.\n\nInstall now?`
          );
          if (!approveDeps) {
            store.setExecutionState('idle');
            store.setExecutionError('Execution cancelled because download dependency installation was declined.');
            store.setDownloadStats(null);
            return;
          }
          store.setExecutionState('installing');
          appendLog(`Installing download dependencies: ${missingDownloadDeps.join(', ')}`);
          await invoke('install_packages', {
            packages: missingDownloadDeps,
            modelId: options.modelId,
            venvModelId: executionEnvModelId,
          });
          appendLog('Download dependency installation complete.');
        }

        const downloaded = await invoke<boolean>('is_model_downloaded', {
          modelId: options.modelId,
          storagePath: options.storagePath,
        });

        if (!downloaded) {
          const shouldDownload = await confirmDialog(
            `Model files are not downloaded yet.\n\nDownload now to:\n${options.storagePath}\n\nYou can pause by clicking Stop and resume later.`
          );
          if (!shouldDownload) {
            store.setExecutionState('idle');
            store.setExecutionError('Execution cancelled because model files were not downloaded.');
            store.setDownloadStats(null);
            return;
          }

          store.setExecutionState('downloading');
          appendLog(`Starting model download for ${options.modelId}`);
          await invoke('download_model', {
            modelId: options.modelId,
            storagePath: options.storagePath,
            hfToken: options.hfToken || null,
          });
          const refreshed = await listDownloadedModels(options.storagePath);
          store.setDownloadedModels(refreshed);
          appendLog('Download stage complete.');
          store.setDownloadStats({
            percent: 100,
            downloadedBytes: 0,
            totalBytes: 0,
            speedBps: 0,
            etaSeconds: 0,
            phase: 'complete',
            filesDone: 0,
            filesTotal: 0,
          });
        }
      } catch (err) {
        stopTimer();
        store.setExecutionState('error');
        store.setExecutionError(String(err));
        store.setDownloadStats(null);
        appendLog(`Preparation failed: ${String(err)}`);
        return;
      }
    }

    if (options?.runMode !== 'direct') {
      const runtimeType = detectRuntimeTypeFromCode(options?.runtimeSourceCode ?? store.generatedCode);
      const prefersCudaForOnnx =
        runtimeType === 'onnxruntime' &&
        options?.preferredDevice !== 'cpu' &&
        Boolean(store.systemInfo.gpuName);
      const runtimePackages = uniqueDependencies([
        ...runtimeDependenciesForType(runtimeType).map((pkg) =>
          runtimeType === 'onnxruntime' && pkg === 'onnxruntime' && prefersCudaForOnnx
            ? 'onnxruntime-gpu'
            : pkg
        ),
        ...(runtimeType === 'transformers_multimodal' && options?.multimodalTask === 'document-understanding'
          ? ['pymupdf']
          : []),
      ]);
      if (runtimePackages.length > 0) {
        appendLog(`Detected runtime type: ${runtimeType}`);
        appendLog(`Checking runtime dependencies before probe: ${runtimePackages.join(', ')}`);
        try {
          const missingRuntime = uniqueDependencies(await invoke<string[]>('check_packages', {
            packages: runtimePackages,
            modelId: options?.modelId ?? null,
            venvModelId: executionEnvModelId,
          }));
          if (missingRuntime.length > 0) {
            if (!allowAutoInstall) {
              stopTimer();
              store.setExecutionState('error');
              store.setExecutionError(
                `Runtime dependencies missing (${missingRuntime.join(', ')}). Auto installs are disabled after first run for this model. Install manually from the terminal panel.`
              );
              store.setDownloadStats(null);
              appendLog(`Missing runtime dependencies with auto-install disabled: ${missingRuntime.join(', ')}`);
              return;
            }

            const approved = await confirmDialog(
              `Runtime dependencies are required before dependency probing:\n\n${missingRuntime.join(', ')}\n\nInstall now?`
            );
            if (!approved) {
              stopTimer();
              store.setExecutionState('idle');
              store.setExecutionError(
                `Execution cancelled. Runtime dependencies were not installed: ${missingRuntime.join(', ')}`
              );
              store.setDownloadStats(null);
              return;
            }

            store.setExecutionState('installing');
            appendLog(`Installing runtime dependencies before probe: ${missingRuntime.join(', ')}`);
            await invoke('install_packages', {
              packages: missingRuntime,
              modelId: options?.modelId ?? null,
              venvModelId: executionEnvModelId,
            });
            appendLog('Runtime dependency installation complete.');
          }
        } catch (err) {
          stopTimer();
          store.setExecutionState('error');
          store.setExecutionError(`Runtime dependency installation failed: ${String(err)}`);
          store.setDownloadStats(null);
          appendLog(`Runtime dependency installation failed: ${String(err)}`);
          return;
        }
      }

      const packages: string[] = [];
      let requiredFromProbe: string[] = [];
      let compatibilityWarning: string | null = null;

      if (options?.modelId) {
        try {
          appendLog('Running dependency compatibility probe.');
          const probe = await invoke<ModelDependencyProbeResult>('probe_model_dependencies', {
            modelId: options.modelId,
            hfToken: options?.hfToken ?? null,
            venvModelId: executionEnvModelId,
          });

          const missingFromProbe = uniqueDependencies(probe.missingPackages ?? []);
          if (missingFromProbe.length > 0) {
            packages.push(...missingFromProbe);
            appendLog(`Model probe found missing imports: ${missingFromProbe.join(', ')}`);
          }

          requiredFromProbe = uniqueRequirementSpecs(probe.requiredPackages ?? []);
          if (requiredFromProbe.length > 0) {
            appendLog(`Model-declared requirements: ${requiredFromProbe.join(', ')}`);
          }

          if (probe.compatibilityError) {
            compatibilityWarning = probe.compatibilityError;
            appendLog(`Compatibility warning: ${probe.compatibilityError}`);
          }
        } catch (err) {
          const probeError = String(err);
          if (/No module named ['"]?transformers['"]?/i.test(probeError)) {
            stopTimer();
            store.setExecutionState('error');
            store.setExecutionError(
              'Dependency probe failed because transformers is missing from the selected environment.'
            );
            store.setDownloadStats(null);
            appendLog(`Dependency probe failed due to missing transformers: ${probeError}`);
            return;
          }
          appendLog(`Dependency probe failed (continuing): ${probeError}`);
        }
      }

      if (compatibilityWarning && requiredFromProbe.length > 0) {
        const requirementsToInstall = requiredFromProbe.filter((req) => hasVersionSpecifier(req));
        if (requirementsToInstall.length > 0) {
          if (!allowAutoInstall) {
            appendLog(
              `Skipping requirement alignment because auto-install is disabled: ${requirementsToInstall.join(', ')}`
            );
          } else {
            const approved = await confirmDialog(
              `Model compatibility warning detected.\n\nInstall model-declared versioned requirements now?\n\n${requirementsToInstall.join('\n')}`
            );
              if (!approved) {
                store.setExecutionState('idle');
                store.setExecutionError(
                  `Execution cancelled. Model requires version-specific dependencies: ${requirementsToInstall.join(', ')}`
                );
                store.setDownloadStats(null);
                return;
              }
            try {
              store.setExecutionState('installing');
              appendLog(`Aligning environment to model requirements: ${requirementsToInstall.join(', ')}`);
              await invoke('install_packages', {
                packages: requirementsToInstall,
                modelId: options?.modelId ?? null,
                venvModelId: executionEnvModelId,
              });
              appendLog('Requirement alignment complete.');
            } catch (err) {
              stopTimer();
              store.setExecutionState('error');
              store.setExecutionError(`Requirement alignment failed: ${String(err)}`);
              store.setDownloadStats(null);
              appendLog(`Requirement alignment failed: ${String(err)}`);
              return;
            }
          }
        }
      }

      if (requiredFromProbe.length > 0) {
        try {
          const missingRequiredNames = uniqueDependencies(await invoke<string[]>('check_packages', {
            packages: uniqueDependencies(requiredFromProbe.map(requirementPackageName)),
            modelId: options?.modelId ?? null,
            venvModelId: executionEnvModelId,
          }));
          const missingRequiredSpecs = filterRequirementSpecsByMissing(requiredFromProbe, missingRequiredNames);

          if (missingRequiredSpecs.length > 0) {
            if (!allowAutoInstall) {
              appendLog(
                `Skipping model-declared requirement install because auto-install is disabled: ${missingRequiredSpecs.join(', ')}`
              );
            } else {
              const approved = await confirmDialog(
                `Model-declared Python requirements were found:\n\n${missingRequiredSpecs.join('\n')}\n\nInstall now?`
              );
              if (!approved) {
                store.setExecutionState('idle');
                store.setExecutionError(
                  `Execution cancelled. Model-declared requirements were not installed: ${missingRequiredSpecs.join(', ')}`
                );
                store.setDownloadStats(null);
                return;
              }

              store.setExecutionState('installing');
              appendLog(`Installing model-declared requirements: ${missingRequiredSpecs.join(', ')}`);
              await invoke('install_packages', {
                packages: missingRequiredSpecs,
                modelId: options?.modelId ?? null,
                venvModelId: executionEnvModelId,
              });
              appendLog('Model-declared requirement installation complete.');
            }
          }
        } catch (err) {
          stopTimer();
          store.setExecutionState('error');
          store.setExecutionError(`Model-declared requirement installation failed: ${String(err)}`);
          store.setDownloadStats(null);
          appendLog(`Model-declared requirement installation failed: ${String(err)}`);
          return;
        }
      }

      if (packages.length > 0) {
        try {
          const missing = uniqueDependencies(await invoke<string[]>('check_packages', {
            packages,
            modelId: options?.modelId ?? null,
            venvModelId: executionEnvModelId,
          }));
          if (missing.length > 0) {
            if (!allowAutoInstall) {
              appendLog(`Skipping package install (auto disabled). Missing: ${missing.join(', ')}`);
            } else {
              const approved = await confirmDialog(
                `Missing packages detected:\n\n${missing.join(', ')}\n\nInstall now?`
              );
              if (!approved) {
                store.setExecutionState('idle');
                store.setExecutionError(
                  `Execution cancelled. Required packages were not installed: ${missing.join(', ')}`
                );
                store.setDownloadStats(null);
                return;
              }
              store.setExecutionState('installing');
              appendLog(`Installing missing packages: ${missing.join(', ')}`);
              await invoke('install_packages', {
                packages: missing,
                modelId: options?.modelId ?? null,
                venvModelId: executionEnvModelId,
              });
              appendLog('Package installation complete.');
            }
          }
        } catch (err) {
          stopTimer();
          store.setExecutionState('error');
          store.setExecutionError(`Package installation failed: ${String(err)}`);
          store.setDownloadStats(null);
          appendLog(`Package installation failed: ${String(err)}`);
          return;
        }
      }
    } else {
      appendLog('Skipping runtime dependency installation and compatibility probe.');
    }

    store.setExecutionState('running');

    appendLog(
      `Launching runtime with device=${options?.preferredDevice ?? 'auto'}${
        options?.selectedGpuId ? `, gpu=${options.selectedGpuId}` : ''
      }${options?.envStoragePath ? `, envStorage=${options.envStoragePath}` : ''}${
        executionEnvModelId ? `, envModel=${executionEnvModelId}` : ''
      }${options?.scriptRelativePath ? `, script=${options.scriptRelativePath}` : ''
      }`
    );

    try {
      await invoke('run_python_code', {
        preferredDevice: options?.preferredDevice ?? 'auto',
        selectedGpuId: options?.selectedGpuId ?? null,
        modelId: options?.modelId ?? null,
        venvModelId: executionEnvModelId,
        hfToken: options?.hfToken ?? null,
        userInput: options?.userInput ?? null,
        envStoragePath: options?.envStoragePath || null,
        storagePath: options?.storagePath || null,
        scriptRelativePath: options?.scriptRelativePath || null,
        diffusionMode: options?.diffusionMode ?? null,
        outputDir: options?.outputDir ?? null,
        negativePrompt: options?.negativePrompt ?? null,
        steps: options?.steps ?? null,
        guidanceScale: options?.guidanceScale ?? null,
        seed: options?.seed ?? null,
        numImages: options?.numImages ?? null,
        strength: options?.strength ?? null,
        sourceImagePath: options?.sourceImagePath ?? null,
        maskImagePath: options?.maskImagePath ?? null,
        multimodalTask: options?.multimodalTask ?? null,
        imagePath: options?.imagePath ?? null,
        documentPath: options?.documentPath ?? null,
      });
    } catch (err) {
      stopTimer();
      store.setExecutionState('error');
      store.setExecutionError(String(err));
      store.setDownloadStats(null);
      appendLog(`Failed to launch execution: ${String(err)}`);
    }
  }, []);

  const cancelExecution = useCallback(async () => {
    try {
      await invoke('cancel_execution');
      await invoke('cancel_download');
      appendLog('Cancellation requested.');
    } catch {
      // Ignore — process may have already finished
    }
    stopTimer();
    flushBufferedStreams();
    useAppStore.getState().setExecutionState('cancelled');
    useAppStore.getState().setDownloadStats(null);
  }, []);

  return { runCode, cancelExecution };
}
