import { create } from 'zustand';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HFModel {
  id: string;
  modelId: string;
  pipeline_tag: string | null;
  tags: string[];
  downloads: number;
  lastModified: string;
  author: string;
  cardData?: {
    license?: string;
  };
  siblings?: Array<{
    rfilename: string;
    size?: number;
    lfs?: { size?: number | null };
  }>;
  description?: string;
}

export interface HFModelDetail extends HFModel {
  readme?: string;
  safetensors?: { total: number };
}

export interface DownloadedModel {
  id: string;
  name: string;
  pipeline_tag: string;
  sizeBytes: number;
  lastUsed: string;
  storagePath: string;
}

export type OutputType =
  | 'text'
  | 'classification'
  | 'image'
  | 'audio'
  | 'embedding'
  | 'none';

export type ExecutionState =
  | 'idle'
  | 'installing'
  | 'downloading'
  | 'running'
  | 'completed'
  | 'error'
  | 'cancelled';

export interface SystemInfo {
  totalRam: number;
  availableRam: number;
  gpuName: string | null;
  gpuVram: number | null;
  os: string;
  pythonReady: boolean;
}

export interface AppSettings {
  hfToken: string;
  claudeApiKey: string;
  modelStoragePath: string;
  envStoragePath: string;
  preferredDevice: 'auto' | 'cpu' | 'cuda';
  selectedGpuId: string | null;
  codeGenerationProvider: 'autorunner' | 'claude-sonnet';
  claudeAutoInstallDependencies: boolean;
  theme: 'dark' | 'light';
}

export interface DownloadStats {
  percent: number;
  downloadedBytes: number;
  totalBytes: number;
  speedBps: number;
  etaSeconds: number | null;
  phase: string;
  filesDone: number;
  filesTotal: number;
  filename?: string;
}

export interface ExecutionStats {
  executionRuntime: string | null;
  executionProvider: string | null;
  processRssBytes: number | null;
  processVirtualMemoryBytes: number | null;
  processGpuBytes: number | null;
  tokensPerSecond: number | null;
  inferenceSeconds: number | null;
}

const MAX_OUTPUT_CHARS = 400_000;

function trimOutputBuffer(current: string, incoming: string): string {
  const merged = current + incoming;
  if (merged.length <= MAX_OUTPUT_CHARS) return merged;
  const trimmed = merged.slice(-MAX_OUTPUT_CHARS);
  return `[HuggingBox] Output trimmed to last ${MAX_OUTPUT_CHARS.toLocaleString()} characters.\n${trimmed}`;
}

// ─── Store Interface ─────────────────────────────────────────────────────────

interface AppStore {
  // Navigation
  currentView: 'browse' | 'model-detail' | 'my-models' | 'settings';
  selectedModelId: string | null;
  activeExecutionModelId: string | null;
  activeExecutionEnvModelId: string | null;

  // Browse
  searchQuery: string;
  pipelineFilter: string | null;
  sizeFilter: string | null;
  sortFilter: 'trendingScore' | 'downloads' | 'likes' | 'lastModified';
  models: HFModel[];
  modelsPage: number;
  modelsHasMore: boolean;
  modelsLoading: boolean;
  modelsError: string | null;
  browseScrollPosition: number;

  // Model Detail
  modelDetail: HFModelDetail | null;
  modelDetailLoading: boolean;
  modelDetailError: string | null;
  /** GGUF file (quantization) chosen on the detail page; null = auto-pick. */
  selectedGguf: string | null;
  generatedCode: string | null;
  codeGenerating: boolean;
  codeSource: 'generated' | 'cached' | 'edited' | null;

  // Execution
  executionState: ExecutionState;
  executionOutput: string;
  stderrOutput: string;
  executionError: string | null;
  executionStartTime: number | null;
  executionElapsed: number;
  downloadStats: DownloadStats | null;
  executionStats: ExecutionStats;

  // Output
  outputType: OutputType;
  outputData: unknown;

  // System
  systemInfo: SystemInfo;

  // Settings
  settings: AppSettings;

  // Downloaded models
  downloadedModels: DownloadedModel[];

  // ─── Actions ───────────────────────────────────────────────────────────────

  setCurrentView: (view: AppStore['currentView']) => void;
  setSelectedModelId: (id: string | null) => void;
  navigateToModel: (id: string, options?: { preserveWorkspace?: boolean }) => void;
  navigateToBrowse: () => void;
  setActiveExecutionModelId: (id: string | null) => void;
  setActiveExecutionEnvModelId: (id: string | null) => void;

  setSearchQuery: (q: string) => void;
  setPipelineFilter: (f: string | null) => void;
  setSizeFilter: (f: string | null) => void;
  setSortFilter: (f: AppStore['sortFilter']) => void;
  setModels: (models: HFModel[]) => void;
  appendModels: (models: HFModel[]) => void;
  setModelsPage: (page: number) => void;
  setModelsHasMore: (has: boolean) => void;
  setModelsLoading: (loading: boolean) => void;
  setModelsError: (err: string | null) => void;
  setBrowseScrollPosition: (pos: number) => void;

  setModelDetail: (detail: HFModelDetail | null) => void;
  setModelDetailLoading: (loading: boolean) => void;
  setModelDetailError: (err: string | null) => void;
  setSelectedGguf: (file: string | null) => void;
  setGeneratedCode: (code: string | null) => void;
  setCodeGenerating: (gen: boolean) => void;
  setCodeSource: (src: AppStore['codeSource']) => void;

  setExecutionState: (state: ExecutionState) => void;
  appendExecutionOutput: (text: string) => void;
  clearExecutionOutput: () => void;
  appendStderrOutput: (text: string) => void;
  clearStderrOutput: () => void;
  setExecutionError: (err: string | null) => void;
  setExecutionStartTime: (t: number | null) => void;
  setExecutionElapsed: (t: number) => void;
  setDownloadStats: (stats: DownloadStats | null) => void;
  setExecutionStats: (stats: Partial<ExecutionStats>) => void;
  clearExecutionStats: () => void;

  setSystemInfo: (info: Partial<SystemInfo>) => void;
  updateSettings: (patch: Partial<AppSettings>) => void;
  setTheme: (theme: 'dark' | 'light') => void;

  setDownloadedModels: (models: DownloadedModel[]) => void;
}

// ─── Store Implementation ────────────────────────────────────────────────────

export const useAppStore = create<AppStore>((set) => ({
  // Navigation
  currentView: 'browse',
  selectedModelId: null,
  activeExecutionModelId: null,
  activeExecutionEnvModelId: null,

  // Browse
  searchQuery: '',
  pipelineFilter: null,
  sizeFilter: null,
  sortFilter: 'trendingScore',
  models: [],
  modelsPage: 0,
  modelsHasMore: true,
  modelsLoading: false,
  modelsError: null,
  browseScrollPosition: 0,

  // Model Detail
  modelDetail: null,
  modelDetailLoading: false,
  modelDetailError: null,
  selectedGguf: null,
  generatedCode: null,
  codeGenerating: false,
  codeSource: null,

  // Execution
  executionState: 'idle',
  executionOutput: '',
  stderrOutput: '',
  executionError: null,
  executionStartTime: null,
  executionElapsed: 0,
  downloadStats: null,
  executionStats: {
    executionRuntime: null,
    executionProvider: null,
    processRssBytes: null,
    processVirtualMemoryBytes: null,
    processGpuBytes: null,
    tokensPerSecond: null,
    inferenceSeconds: null,
  },

  // Output
  outputType: 'none',
  outputData: null,

  // System
  systemInfo: {
    totalRam: 0,
    availableRam: 0,
    gpuName: null,
    gpuVram: null,
    os: '',
    pythonReady: false,
  },

  // Settings
  settings: {
    hfToken: '',
    claudeApiKey: '',
    modelStoragePath: '~/HuggingBox/models/',
    envStoragePath: '',
    preferredDevice: 'auto',
    selectedGpuId: null,
    codeGenerationProvider: 'autorunner',
    claudeAutoInstallDependencies: false,
    theme: 'dark',
  },

  downloadedModels: [],

  // ─── Action implementations ────────────────────────────────────────────────

  setCurrentView: (view) => set({ currentView: view }),
  setSelectedModelId: (id) => set({ selectedModelId: id }),
  navigateToModel: (id, options) =>
    set((s) => {
      if (options?.preserveWorkspace) {
        return {
          selectedModelId: id,
          currentView: 'model-detail',
        };
      }

      return {
        selectedModelId: id,
        currentView: 'model-detail',
        modelDetail: null,
        modelDetailError: null,
        selectedGguf: null,
        generatedCode: null,
        codeSource: null,
        executionState: 'idle',
        executionOutput: '',
        stderrOutput: '',
        executionError: null,
        executionStartTime: null,
        executionElapsed: 0,
        downloadStats: null,
        executionStats: {
          executionRuntime: null,
          executionProvider: null,
          processRssBytes: null,
          processVirtualMemoryBytes: null,
          processGpuBytes: null,
          tokensPerSecond: null,
          inferenceSeconds: null,
        },
        activeExecutionModelId:
          s.activeExecutionModelId && s.activeExecutionModelId === id
            ? s.activeExecutionModelId
            : null,
        activeExecutionEnvModelId:
          s.activeExecutionModelId && s.activeExecutionModelId === id
            ? s.activeExecutionEnvModelId
            : null,
      };
    }),
  navigateToBrowse: () =>
    set({ currentView: 'browse', selectedModelId: null }),
  setActiveExecutionModelId: (id) => set({ activeExecutionModelId: id }),
  setActiveExecutionEnvModelId: (id) => set({ activeExecutionEnvModelId: id }),

  setSearchQuery: (q) => set({ searchQuery: q }),
  setPipelineFilter: (f) => set({ pipelineFilter: f }),
  setSizeFilter: (f) => set({ sizeFilter: f }),
  setSortFilter: (f) => set({ sortFilter: f }),
  setModels: (models) => set({ models }),
  appendModels: (models) =>
    set((s) => ({ models: [...s.models, ...models] })),
  setModelsPage: (page) => set({ modelsPage: page }),
  setModelsHasMore: (has) => set({ modelsHasMore: has }),
  setModelsLoading: (loading) => set({ modelsLoading: loading }),
  setModelsError: (err) => set({ modelsError: err }),
  setBrowseScrollPosition: (pos) => set({ browseScrollPosition: pos }),

  setModelDetail: (detail) => set({ modelDetail: detail }),
  setModelDetailLoading: (loading) => set({ modelDetailLoading: loading }),
  setModelDetailError: (err) => set({ modelDetailError: err }),
  setSelectedGguf: (file) => set({ selectedGguf: file }),
  setGeneratedCode: (code) => set({ generatedCode: code }),
  setCodeGenerating: (gen) => set({ codeGenerating: gen }),
  setCodeSource: (src) => set({ codeSource: src }),

  setExecutionState: (state) => set({ executionState: state }),
  appendExecutionOutput: (text) =>
    set((s) => ({ executionOutput: trimOutputBuffer(s.executionOutput, text) })),
  clearExecutionOutput: () => set({ executionOutput: '' }),
  appendStderrOutput: (text) =>
    set((s) => ({ stderrOutput: trimOutputBuffer(s.stderrOutput, text) })),
  clearStderrOutput: () => set({ stderrOutput: '' }),
  setExecutionError: (err) => set({ executionError: err }),
  setExecutionStartTime: (t) => set({ executionStartTime: t }),
  setExecutionElapsed: (t) => set({ executionElapsed: t }),
  setDownloadStats: (stats) => set({ downloadStats: stats }),
  setExecutionStats: (stats) =>
    set((s) => ({ executionStats: { ...s.executionStats, ...stats } })),
  clearExecutionStats: () =>
    set({
      executionStats: {
        executionRuntime: null,
        executionProvider: null,
        processRssBytes: null,
        processVirtualMemoryBytes: null,
        processGpuBytes: null,
        tokensPerSecond: null,
        inferenceSeconds: null,
      },
    }),

  setSystemInfo: (info) =>
    set((s) => ({ systemInfo: { ...s.systemInfo, ...info } })),
  updateSettings: (patch) =>
    set((s) => {
      const newSettings = { ...s.settings, ...patch };
      if (
        patch.claudeApiKey !== undefined &&
        patch.claudeApiKey.trim() !== '' &&
        s.settings.codeGenerationProvider === 'autorunner'
      ) {
        newSettings.codeGenerationProvider = 'claude-sonnet';
      }
      return { settings: newSettings };
    }),
  setTheme: (theme) =>
    set((s) => ({ settings: { ...s.settings, theme } })),

  setDownloadedModels: (models) => set({ downloadedModels: models }),
}));
