import { useEffect, useRef, useCallback } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../stores/appStore';
import { listDownloadedModels } from '../services/modelStorage';
import { listModelEnvironments } from '../services/modelEnvironments';
import { confirmDialog } from '../services/dialogs';
import {
  getPreferredEnvModelId,
  setPreferredEnvModelId,
} from '../services/modelEnvPreference';

interface ModelDependencyProbeResult {
  missingPackages: string[];
  requiredPackages?: string[];
  compatibilityError?: string | null;
}

interface ExecutionStatsPayload {
  job_id?: string;
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
    return ['torch', 'transformers', 'pillow'];
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
  return ['torch', 'transformers', 'pillow'];
}

let activePreparation: {id: string; cancelled: boolean; settled: Promise<void>} | null = null;
const PREPARED_KEY = 'huggingbox-prepared-v1';
const prepared = new Set<string>();
try { const saved: unknown = JSON.parse(localStorage.getItem(PREPARED_KEY) || '[]'); if (Array.isArray(saved)) for (const item of saved) if (typeof item === 'string') prepared.add(item); } catch { /* Corrupt readiness data is safely rebuilt. */ }
function savePrepared() { try { localStorage.setItem(PREPARED_KEY, JSON.stringify([...prepared].slice(-100))); } catch { /* Running does not depend on persistence. */ } }
async function preparationKey(parts: unknown[]): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(parts)));
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
}
let activePreparationKey: string | null = null;
let cancellingJobId: string | null = null;

async function chooseExecutionEnvModelId(modelId: string, envStoragePath?: string, requested?: string): Promise<string> {
  const preferred = requested || getPreferredEnvModelId(modelId) || modelId;
  if (preferred !== modelId) {
    const envs = await listModelEnvironments(envStoragePath);
    if (!envs.some(e => e.modelId === preferred)) throw new Error('The selected environment no longer exists. Select a new isolated environment.');
  }
  setPreferredEnvModelId(modelId, preferred);
  return preferred;
}

function isCurrentEvent(payload: {job_id?: string}): boolean {
  return Boolean(payload.job_id && payload.job_id !== cancellingJobId && payload.job_id === useAppStore.getState().activeJobId);
}

async function ensureExecutionListenersRegistered(): Promise<void> {
  if (executionListenersCleanup) return;
  if (executionListenersInitPromise) {
    await executionListenersInitPromise;
    return;
  }

  executionListenersInitPromise = (async () => {
    const unlistens: Array<() => void> = [];

    const u1 = await listen<{ text: string; job_id?: string }>('execution-stdout', (e) => {
      if (!isCurrentEvent(e.payload)) return;
      queueStdout(e.payload.text);
    });

    const u2 = await listen<{ text: string; job_id?: string }>('execution-stderr', (e) => {
      if (!isCurrentEvent(e.payload)) return;
      queueStderr(e.payload.text);
    });

    const u3 = await listen<{ exit_code: number; job_id?: string }>('execution-done', (e) => {
      if (!isCurrentEvent(e.payload)) return;
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
      appendLog(`Execution finished with exit code ${e.payload.exit_code}.`);
      if (state === 'error' && activePreparationKey) { prepared.delete(activePreparationKey); savePrepared(); }
      const started = useAppStore.getState().executionStartTime;
      if (started) useAppStore.getState().setExecutionElapsed(Date.now() - started);
      useAppStore.getState().finishRun(state);
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
      if (!isCurrentEvent(e.payload)) return;
      useAppStore.getState().setExecutionStats({
        processRssBytes: e.payload.rss_bytes,
        processVirtualMemoryBytes: e.payload.virtual_memory_bytes,
        processGpuBytes: e.payload.gpu_memory_bytes ?? null,
      });

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
      ggufFile?: string;
      environmentId?: string;
      repair?: boolean;
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
    if (store.activeJobId || activePreparation || cancellingJobId) return;
    const modelId = options?.modelId?.trim();
    if (!modelId || !options?.scriptRelativePath || !options.runtimeSourceCode?.trim()) {
      store.setExecutionError('Prepare a supported model before running. The script is empty.');
      return;
    }
    const job = {id: crypto.randomUUID(), cancelled: false, settled: Promise.resolve()};
    activePreparation = job;
    useAppStore.setState({activeJobId: job.id, activeExecutionModelId: modelId, activeRunInput: options.userInput ?? ''});
    resetBufferedStreams();
    store.clearExecutionOutput(); store.clearStderrOutput(); store.clearExecutionStats(); store.setExecutionElapsed(0);
    store.setExecutionError(null); store.setDownloadStats(null);
    store.setExecutionState('installing'); startTimer();
    const check = () => { if (job.cancelled || useAppStore.getState().activeJobId !== job.id) throw new Error('Run cancelled.'); };
    async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
      check(); const value = await invoke<T>(command, args); check(); return value;
    }
    const perform = async () => {
      try {
        await call('begin_preparation');
        const environment = await chooseExecutionEnvModelId(modelId, options.envStoragePath, options.environmentId);
        check(); store.setActiveExecutionEnvModelId(environment);
        const runtime = detectRuntimeTypeFromCode(options.runtimeSourceCode);
        const context = {modelId, venvModelId: environment, envStoragePath: options.envStoragePath || null};
        const fingerprint = await call<string>('environment_fingerprint', {modelId: environment, envStoragePath: options.envStoragePath || null});
        const identity = [modelId, environment, options.envStoragePath, options.storagePath, options.ggufFile, runtime, options.runtimeSourceCode, options.preferredDevice, options.selectedGpuId];
        const key = await preparationKey([...identity, fingerprint]); check();
        activePreparationKey = key;
        if (options.repair) prepared.delete(key);
        const downloaded = await call<boolean>('is_model_downloaded', {modelId, storagePath: options.storagePath, runtime, ggufFile: options.ggufFile ?? null});
        if (!downloaded) prepared.delete(key);
        if (options.runMode !== 'direct' && !prepared.has(key)) {
          const approve = await confirmDialog('Prepare this model? HuggingBox will install its required packages in the selected environment, download only the selected model files, and verify readiness. Your scripts and results are kept separately from disposable weights.', {okLabel: 'Prepare and run', cancelLabel: 'Cancel'});
          check();
          if (!approve) { job.cancelled = true; throw new Error('Preparation cancelled.'); }
          const dependencies = uniqueDependencies(['huggingface_hub', ...runtimeDependenciesForType(runtime)]);
          const missing = await call<string[]>('check_packages', {...context, packages: dependencies});
          if (missing.length) {
            appendLog(`Installing required packages: ${missing.join(', ')}`);
            await call('install_packages', {...context, packages: missing});
          }
          if (!downloaded) {
            store.setExecutionState('downloading'); appendLog('Downloading selected model files…');
            await call('download_model', {...context, storagePath: options.storagePath, hfToken: options.hfToken || null, runtime, ggufFile: options.ggufFile ?? null});
            const models = await listDownloadedModels(options.storagePath!); check(); store.setDownloadedModels(models);
          }
          store.setExecutionState('installing'); appendLog('Checking model requirements…');
          const probe = await call<ModelDependencyProbeResult>('probe_model_dependencies', {...context, hfToken: options.hfToken || null, storagePath: options.storagePath});
          const requirements = uniqueRequirementSpecs([...(probe.requiredPackages ?? []), ...(probe.missingPackages ?? [])]);
          if (requirements.length) {
            const missingNames = await call<string[]>('check_packages', {...context, packages: requirements});
            const missingSet = new Set(missingNames.map(requirementPackageName));
            const needed = requirements.filter(r => missingSet.has(requirementPackageName(r)) || Boolean(probe.compatibilityError));
            if (needed.length) await call('install_packages', {...context, packages: needed});
          }
          if (probe.compatibilityError || requirements.length) {
            const verified = await call<ModelDependencyProbeResult>('probe_model_dependencies', {...context, hfToken: options.hfToken || null, storagePath: options.storagePath});
            if (verified.compatibilityError || verified.missingPackages.length) throw new Error(verified.compatibilityError || `Missing packages: ${verified.missingPackages.join(', ')}`);
          }
          const readyFingerprint = await call<string>('environment_fingerprint', {modelId: environment, envStoragePath: options.envStoragePath || null});
          activePreparationKey = await preparationKey([...identity, readyFingerprint]); check();
          prepared.add(activePreparationKey); savePrepared(); appendLog('Model prepared. Future runs reuse this setup.');
        } else if (options.runMode === 'direct' && !downloaded) {
          throw new Error('Prepare this model before using direct execution.');
        }
        check(); store.setDownloadStats(null); store.setExecutionState('running');
        await call('run_python_code', {
          jobId: job.id, modelId, venvModelId: environment, storagePath: options.storagePath,
          preferredDevice: options.preferredDevice ?? 'auto', selectedGpuId: options.selectedGpuId ?? null,
          hfToken: options.hfToken ?? null, userInput: options.userInput ?? null,
          envStoragePath: options.envStoragePath || null, scriptRelativePath: options.scriptRelativePath,
          diffusionMode: options.diffusionMode ?? null, outputDir: options.outputDir ?? null,
          negativePrompt: options.negativePrompt ?? null, steps: options.steps ?? null, guidanceScale: options.guidanceScale ?? null,
          seed: options.seed ?? null, numImages: options.numImages ?? null, strength: options.strength ?? null,
          sourceImagePath: options.sourceImagePath ?? null, maskImagePath: options.maskImagePath ?? null,
          multimodalTask: options.multimodalTask ?? null, imagePath: options.imagePath ?? null, documentPath: options.documentPath ?? null,
        });
      } catch (error) {
        if (activePreparationKey) { prepared.delete(activePreparationKey); savePrepared(); }
        if (useAppStore.getState().activeJobId === job.id && cancellingJobId !== job.id) {
          store.setExecutionError(job.cancelled ? null : String(error));
          store.finishRun(job.cancelled ? 'cancelled' : 'error');
        }
        stopTimer();
      } finally {
        if (activePreparation === job) activePreparation = null;
      }
    };
    job.settled = perform();
    await job.settled;
  }, []);

  const cancelExecution = useCallback(async () => {
    const job = activePreparation;
    if (job) job.cancelled = true;
    const id = useAppStore.getState().activeJobId;
    if (cancellingJobId) return;
    cancellingJobId = id;
    if (id) useAppStore.getState().setExecutionState('stopping');
    await Promise.allSettled([invoke('cancel_preparation'), invoke('cancel_execution'), invoke('cancel_download')]);
    if (job) await job.settled;
    stopTimer(); flushBufferedStreams();
    if (id && useAppStore.getState().activeJobId === id) useAppStore.getState().finishRun('cancelled');
    cancellingJobId = null;
    useAppStore.getState().setDownloadStats(null);
  }, []);

  return { runCode, cancelExecution };
}
