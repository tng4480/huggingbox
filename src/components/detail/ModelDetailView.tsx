import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, AlertCircle, RefreshCw } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../../stores/appStore';
import type { HFModelDetail } from '../../stores/appStore';
import { confirmDialog, messageDialog } from '../../services/dialogs';
import { fetchModelDetail, estimateModelSize, formatBytes, getModelFormatInfo, resolveGgufFile } from '../../services/huggingfaceApi';
import { estimateRamBytes } from '../../utils/ramEstimation';
import {
  buildCacheIdentity,
  buildFallbackAnalysis,
  buildFallbackCode,
  fixCodeWithClaude,
  generateCodeWithClaude,
  generateCodeLocally,
} from '../../services/hfAutoRunner';
import {
  buildCodeCacheKey,
  readCachedCode,
  writeCachedCode,
} from '../../services/codeCache';
import ModelInfoPanel from './ModelInfoPanel';
import CodeEditor from './CodeEditor';
import FileExplorer from './FileExplorer';
import InputPanel, { type DiffusionMode, type MultimodalTask } from './InputPanel';
import OutputPanel from './OutputPanel';
import Button from '../shared/Button';
import SkeletonCard from '../shared/SkeletonCard';
import { useExecution } from '../../hooks/useExecution';
import {
  createModelWorkspaceDirectory,
  createModelWorkspaceFile,
  listModelWorkspaceEntries,
  readModelWorkspaceFile,
  type ModelWorkspaceEntry,
  writeModelWorkspaceFile,
} from '../../services/modelWorkspace';

export default function ModelDetailView() {
  const selectedModelId = useAppStore((s) => s.selectedModelId);
  const modelDetail = useAppStore((s) => s.modelDetail);
  const modelDetailLoading = useAppStore((s) => s.modelDetailLoading);
  const modelDetailError = useAppStore((s) => s.modelDetailError);
  const generatedCode = useAppStore((s) => s.generatedCode);
  const codeGenerating = useAppStore((s) => s.codeGenerating);
  const executionState = useAppStore((s) => s.executionState);
  const executionError = useAppStore((s) => s.executionError);
  const stderrOutput = useAppStore((s) => s.stderrOutput);
  const activeExecutionModelId = useAppStore((s) => s.activeExecutionModelId);
  const activeExecutionEnvModelId = useAppStore((s) => s.activeExecutionEnvModelId);
  const navigateToBrowse = useAppStore((s) => s.navigateToBrowse);
  const setModelDetail = useAppStore((s) => s.setModelDetail);
  const setModelDetailLoading = useAppStore((s) => s.setModelDetailLoading);
  const setModelDetailError = useAppStore((s) => s.setModelDetailError);
  const setGeneratedCode = useAppStore((s) => s.setGeneratedCode);
  const setCodeGenerating = useAppStore((s) => s.setCodeGenerating);
  const setCodeSource = useAppStore((s) => s.setCodeSource);
  const codeSource = useAppStore((s) => s.codeSource);
  const settings = useAppStore((s) => s.settings);
  const systemInfo = useAppStore((s) => s.systemInfo);
  const [codeGenerationError, setCodeGenerationError] = useState<string | null>(null);
  const [claudeAnalysis, setClaudeAnalysis] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedModelId) return;
    if (modelDetail?.id === selectedModelId || modelDetail?.modelId === selectedModelId) return;

    async function load() {
      setModelDetailLoading(true);
      setModelDetailError(null);
      try {
        const detail = await fetchModelDetail(
          selectedModelId!,
          settings.hfToken || undefined
        );
        setModelDetail(detail);
      } catch {
        setModelDetailError("Couldn't load model details. Try again.");
      } finally {
        setModelDetailLoading(false);
      }
    }

    load();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId]);

  useEffect(() => {
    setCodeGenerationError(null);
    setClaudeAnalysis(null);
  }, [selectedModelId]);

  async function generateCode(options?: { bypassCache?: boolean }) {
    if (!modelDetail) return;
    setCodeGenerationError(null);

    if (settings.codeGenerationProvider === 'claude-sonnet' && !settings.claudeApiKey.trim()) {
      setCodeGenerationError('Claude Sonnet generation requires an Anthropic API key in Settings.');
      return;
    }

    setCodeGenerating(true);

    const modelId = modelDetail.modelId ?? modelDetail.id;
    const ggufFile = resolveGgufFile(modelDetail, useAppStore.getState().selectedGguf)?.name ?? null;
    const cacheIdentity = `${buildCacheIdentity(modelDetail, systemInfo)}_generator:${settings.codeGenerationProvider}${
      ggufFile ? `_gguf:${ggufFile}` : ''
    }`;
    const bypassCache = options?.bypassCache === true;

    try {
      const cacheKey = await buildCodeCacheKey(cacheIdentity);
      const cached = bypassCache ? null : await readCachedCode(cacheKey);
      if (cached) {
        setGeneratedCode(cached);
        setCodeSource('cached');
        setClaudeAnalysis('Loaded cached code. Click Re-gen to refresh Claude analysis for this model.');
      } else {
        let code: string;
        try {
          const usingClaude = settings.codeGenerationProvider === 'claude-sonnet';
          const generated = usingClaude
            ? await generateCodeWithClaude(modelDetail, settings, systemInfo)
            : await generateCodeLocally(modelDetail, settings, systemInfo, ggufFile);
          code = generated.code;
          setClaudeAnalysis(generated.analysis);
          setCodeSource('generated');

          if (usingClaude && Array.isArray(generated.dependencies) && generated.dependencies.length > 0) {
            const missingClaudeDependencies = await invoke<string[]>('check_packages', {
              packages: generated.dependencies,
              modelId,
              venvModelId: modelId,
            });

            if (missingClaudeDependencies.length === 0) {
              setClaudeAnalysis(
                `${generated.analysis}\n\nClaude-suggested dependencies are already installed in this model environment.`
              );
            } else if (settings.claudeAutoInstallDependencies) {
              const approved = await confirmDialog(
                `Claude suggested these Python dependencies for ${modelId}.\n\nAlready installed dependencies were skipped.\n\nMissing dependencies:\n${missingClaudeDependencies.join('\n')}\n\nInstall the missing dependencies into this model environment now?`
              );
              if (approved) {
                await invoke('install_packages', {
                  packages: missingClaudeDependencies,
                  modelId,
                  venvModelId: modelId,
                });
                setClaudeAnalysis(
                  `${generated.analysis}\n\nInstalled Claude-suggested missing dependencies:\n${missingClaudeDependencies.join(', ')}`
                );
              } else {
                setClaudeAnalysis(
                  `${generated.analysis}\n\nClaude-suggested missing dependencies:\n${missingClaudeDependencies.join(', ')}`
                );
              }
            } else {
              setClaudeAnalysis(
                `${generated.analysis}\n\nClaude-suggested missing dependencies:\n${missingClaudeDependencies.join(', ')}`
              );
            }
          }
        } catch (err) {
          const providerLabel =
            settings.codeGenerationProvider === 'claude-sonnet'
              ? 'Claude Sonnet'
              : 'hf_auto_runner';
          console.error(`${providerLabel} generation failed`, err);
          code = buildFallbackCode(modelDetail);
          setClaudeAnalysis(buildFallbackAnalysis(modelDetail, systemInfo, String(err)));
          setCodeSource('generated');
          setCodeGenerationError(
            `${providerLabel} failed to generate code. Loaded a local fallback template instead. Error: ${String(err)}`
          );
        }

        setGeneratedCode(code);
        await writeCachedCode(cacheKey, modelId, code);
      }
    } catch (err) {
      setCodeGenerationError(String(err));
    } finally {
      setCodeGenerating(false);
    }
  }

  async function handleGenerateCode() {
    await generateCode();
  }

  function handleUseModel() {
    setCodeGenerationError(null);
    setGeneratedCode('');
    setCodeSource(null);
  }

  async function handleRegenerateCode() {
    if (codeSource === 'edited') {
      const proceed = await confirmDialog(
        "You've edited the code. Regenerating will replace your changes. Continue?"
      );
      if (!proceed) return;
    }
    await generateCode({ bypassCache: true });
  }

  async function syncClaudeDependencies(
    modelId: string,
    dependencies: string[] | undefined,
    analysis: string
  ): Promise<string> {
    if (!Array.isArray(dependencies) || dependencies.length === 0) {
      return analysis;
    }

    const missingClaudeDependencies = await invoke<string[]>('check_packages', {
      packages: dependencies,
      modelId,
      venvModelId: modelId,
    });

    if (missingClaudeDependencies.length === 0) {
      return `${analysis}\n\nClaude-suggested dependencies are already installed in this model environment.`;
    }

    if (!settings.claudeAutoInstallDependencies) {
      return `${analysis}\n\nClaude-suggested missing dependencies:\n${missingClaudeDependencies.join(', ')}`;
    }

    const approved = await confirmDialog(
      `Claude suggested these Python dependencies for ${modelId}.\n\nAlready installed dependencies were skipped.\n\nMissing dependencies:\n${missingClaudeDependencies.join('\n')}\n\nInstall the missing dependencies into this model environment now?`
    );
    if (!approved) {
      return `${analysis}\n\nClaude-suggested missing dependencies:\n${missingClaudeDependencies.join(', ')}`;
    }

    await invoke('install_packages', {
      packages: missingClaudeDependencies,
      modelId,
      venvModelId: modelId,
    });
    return `${analysis}\n\nInstalled Claude-suggested missing dependencies:\n${missingClaudeDependencies.join(', ')}`;
  }

  async function handleFixCode() {
    if (!modelDetail) return;
    const modelId = modelDetail.modelId ?? modelDetail.id;
    const currentCode = generatedCode ?? '';
    const envModelId =
      activeExecutionModelId === modelId && activeExecutionEnvModelId
        ? activeExecutionEnvModelId
        : modelId;

    if (!settings.claudeApiKey.trim()) {
      setCodeGenerationError('Claude Sonnet fix mode requires an Anthropic API key in Settings.');
      return;
    }

    setCodeGenerationError(null);
    setCodeGenerating(true);

    try {
      const freezeResult = await invoke<{
        stdout: string;
        stderr: string;
        exitCode: number;
        cwd: string;
      }>('run_model_shell_command', {
        modelId,
        venvModelId: envModelId,
        command: 'python -m pip freeze',
        modelStoragePath: settings.modelStoragePath || null,
        envStoragePath: settings.envStoragePath || null,
        cwd: null,
        hfToken: settings.hfToken || null,
      });

      const fixed = await fixCodeWithClaude(modelDetail, settings, systemInfo, {
        currentCode,
        stderrOutput: stderrOutput || freezeResult.stderr || '',
        installedModules: freezeResult.stdout || '',
        executionError,
      });

      setGeneratedCode(fixed.code);
      setCodeSource('generated');
      const enrichedAnalysis = await syncClaudeDependencies(modelId, fixed.dependencies, fixed.analysis);
      setClaudeAnalysis(enrichedAnalysis);
      setCodeGenerationError(null);
    } catch (error) {
      setCodeGenerationError(`Claude fix failed: ${String(error)}`);
    } finally {
      setCodeGenerating(false);
    }
  }

  const showFixButton =
    generatedCode !== null &&
    (executionState === 'error' || Boolean(executionError?.trim()) || Boolean(stderrOutput.trim()));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Back nav bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 var(--space-xl)',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
          backgroundColor: 'var(--bg-primary)',
          height: '44px',
        }}
      >
        <Button
          variant="ghost"
          icon={<ArrowLeft size={16} strokeWidth={1.5} />}
          onClick={navigateToBrowse}
          aria-label="Back to Browse"
          style={{ color: 'var(--text-secondary)', padding: '0 var(--space-sm)' }}
        >
          Back to Browse
        </Button>

        {modelDetail && generatedCode !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <span
              style={{
                fontFamily: '"Inter", sans-serif',
                fontSize: '14px',
                fontWeight: 600,
                color: 'var(--text-primary)',
                maxWidth: '300px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {selectedModelId}
            </span>
            <Button
              variant="secondary"
              onClick={generatedCode.trim() ? handleRegenerateCode : handleGenerateCode}
              disabled={codeGenerating}
              style={{ fontSize: '12px', height: '28px', padding: '0 var(--space-md)' }}
            >
              {generatedCode.trim() ? 'Re-gen' : 'Generate Code'}
            </Button>
            {showFixButton && (
              <Button
                variant="secondary"
                onClick={handleFixCode}
                disabled={codeGenerating}
                style={{ fontSize: '12px', height: '28px', padding: '0 var(--space-md)' }}
              >
                Fix
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Loading skeleton */}
        {modelDetailLoading && (
          <div
            style={{
              padding: 'var(--space-2xl)',
              maxWidth: '800px',
              margin: '0 auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-lg)',
              width: '100%',
            }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              <div className="skeleton" style={{ height: '20px', width: '80px' }} />
              <div className="skeleton" style={{ height: '28px', width: '60%' }} />
            </div>
            <div className="skeleton" style={{ height: '60px', width: '100%' }} />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
              {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
            </div>
          </div>
        )}

        {/* Error state */}
        {modelDetailError && !modelDetailLoading && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 'var(--space-lg)',
              padding: 'var(--space-3xl)',
              color: 'var(--text-muted)',
              fontFamily: '"Inter", sans-serif',
              fontSize: '14px',
            }}
          >
            <AlertCircle size={32} strokeWidth={1.5} color="var(--error)" />
            <p>{modelDetailError}</p>
            <Button
              variant="secondary"
              icon={<RefreshCw size={14} strokeWidth={1.5} />}
              onClick={() => {
                setModelDetailError(null);
                setModelDetail(null);
              }}
            >
              Retry
            </Button>
          </div>
        )}

        {/* Phase 1: Model info + Generate Code button */}
        {!modelDetailLoading && !modelDetailError && modelDetail && generatedCode === null && (
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <ModelInfoPanel
              model={modelDetail}
              onUseModel={handleUseModel}
              codeGenerationError={codeGenerationError}
            />
          </div>
        )}

        {/* Phase 2: Three-panel workspace */}
        {!modelDetailLoading && !modelDetailError && modelDetail && generatedCode !== null && (
          <WorkspaceLayout
            model={modelDetail}
            code={generatedCode}
            executionState={executionState}
            claudeAnalysis={claudeAnalysis}
            codeGenerationError={codeGenerationError}
          />
        )}
      </div>
    </div>
  );
}

// ─── Phase 2: Three-panel Workspace ──────────────────────────────────────────

interface WorkspaceLayoutProps {
  model: HFModelDetail;
  code: string;
  executionState: string;
  claudeAnalysis: string | null;
  codeGenerationError: string | null;
}

function WorkspaceLayout({
  model,
  code,
  executionState,
  claudeAnalysis,
  codeGenerationError,
}: WorkspaceLayoutProps) {
  const [inputValue, setInputValue] = useState('');
  const [runMode, setRunMode] = useState<'prepared' | 'direct'>('prepared');
  const [diffusionMode, setDiffusionMode] = useState<DiffusionMode>('text-to-image');
  const [sourceImagePath, setSourceImagePath] = useState('');
  const [maskImagePath, setMaskImagePath] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState('');
  const [numImages, setNumImages] = useState(1);
  const [strength, setStrength] = useState(0.75);
  const [multimodalTask, setMultimodalTask] = useState<MultimodalTask>('visual-question-answering');
  const [multimodalImagePath, setMultimodalImagePath] = useState('');
  const [multimodalDocumentPath, setMultimodalDocumentPath] = useState('');
  const [currentDirectory, setCurrentDirectory] = useState('');
  const [workspaceEntries, setWorkspaceEntries] = useState<ModelWorkspaceEntry[]>([]);
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [editorCode, setEditorCode] = useState(code);
  const [editorStatus, setEditorStatus] = useState('Python');
  const initialCodeRef = useRef(code);
  const skipNextAutosaveRef = useRef(true);
  const previousExecutionStateRef = useRef(executionState);
  const settings = useAppStore((s) => s.settings);
  const setGeneratedCode = useAppStore((s) => s.setGeneratedCode);
  const setCodeSource = useAppStore((s) => s.setCodeSource);
  const { runCode, cancelExecution } = useExecution();
  const modelId = useMemo(() => model.modelId ?? model.id, [model.id, model.modelId]);
  const supportsDiffusionModes = useMemo(() => {
    const pipeline = (model.pipeline_tag ?? '').toLowerCase();
    if (['text-to-image', 'image-to-image', 'inpainting'].includes(pipeline)) return true;
    return /(^|\n)\s*#?\s*RUNTIME:\s*diffusers\b/i.test(editorCode);
  }, [editorCode, model.pipeline_tag]);
  const supportsMultimodalModes = useMemo(() => {
    const pipeline = (model.pipeline_tag ?? '').toLowerCase();
    if (
      ['image-text-to-text', 'image-to-text', 'visual-question-answering', 'document-question-answering'].includes(
        pipeline
      )
    ) {
      return true;
    }
    return /(^|\n)\s*#?\s*RUNTIME:\s*transformers_multimodal\b/i.test(editorCode);
  }, [editorCode, model.pipeline_tag]);

  const isRunning =
    executionState === 'running' ||
    executionState === 'installing' ||
    executionState === 'downloading';

  // Derive metadata labels for InputPanel info section
  const selectedGguf = useAppStore((s) => s.selectedGguf);
  const sizeBytes = estimateModelSize(model, selectedGguf);
  const sizeLabel = sizeBytes > 0 ? formatBytes(sizeBytes) : undefined;

  const ramBytes = estimateRamBytes(model, selectedGguf);
  const ramLabel = ramBytes > 0 ? `~${formatBytes(ramBytes)}` : undefined;

  const formatInfo = getModelFormatInfo(model);
  const formatLabel =
    (formatInfo.hasGguf && 'GGUF') ||
    (formatInfo.hasOnnx && 'ONNX') ||
    (formatInfo.hasSafetensors && 'SafeTensors') ||
    (formatInfo.hasPytorch && 'PyTorch') ||
    undefined;

  useEffect(() => {
    initialCodeRef.current = code;
  }, [code, modelId]);

  useEffect(() => {
    if (code === editorCode) return;
    setEditorCode(code);
    skipNextAutosaveRef.current = true;
  }, [code]);

  useEffect(() => {
    const pipeline = (model.pipeline_tag ?? '').toLowerCase();
    if (pipeline === 'image-to-image') {
      setDiffusionMode('image-to-image');
    } else if (pipeline === 'inpainting') {
      setDiffusionMode('inpainting');
    } else if (pipeline === 'text-to-image') {
      setDiffusionMode('text-to-image');
    }
  }, [model.pipeline_tag, modelId]);

  useEffect(() => {
    const pipeline = (model.pipeline_tag ?? '').toLowerCase();
    if (pipeline === 'document-question-answering') {
      setMultimodalTask('document-understanding');
    } else if (pipeline === 'image-to-text') {
      setMultimodalTask('image-captioning');
    } else {
      setMultimodalTask('visual-question-answering');
    }
  }, [model.pipeline_tag, modelId]);

  function normalizeRelativePath(input: string): string {
    return input
      .trim()
      .replace(/\\/g, '/')
      .replace(/^\/+/, '')
      .split('/')
      .map((segment) => segment.trim())
      .filter((segment) => segment.length > 0 && segment !== '.')
      .join('/');
  }

  function parentDirectory(path: string): string {
    const parts = path.split('/').filter(Boolean);
    if (parts.length <= 1) return '';
    parts.pop();
    return parts.join('/');
  }

  function fileName(path: string | null): string {
    if (!path) return 'huggingbox_main.py';
    const parts = path.split('/').filter(Boolean);
    return parts[parts.length - 1] || path;
  }

  const refreshWorkspaceEntries = useCallback(
    async (directory: string) => {
      const rows = await listModelWorkspaceEntries(modelId, settings.modelStoragePath, directory);
      setWorkspaceEntries(rows);
    },
    [modelId, settings.modelStoragePath]
  );

  useEffect(() => {
    const previous = previousExecutionStateRef.current;
    previousExecutionStateRef.current = executionState;
    if (
      previous === executionState ||
      !['running', 'installing', 'downloading'].includes(previous) ||
      !['completed', 'error', 'cancelled'].includes(executionState)
    ) {
      return;
    }

    void refreshWorkspaceEntries(currentDirectory).catch(() => {
      // leave the current explorer state alone if refresh fails
    });
  }, [currentDirectory, executionState, refreshWorkspaceEntries]);

  const openWorkspaceFile = useCallback(
    async (relativePath: string) => {
      const content = await readModelWorkspaceFile(modelId, settings.modelStoragePath, relativePath);
      setSelectedFilePath(relativePath);
      setEditorCode(content);
      setGeneratedCode(content);
      setCodeSource('edited');
      skipNextAutosaveRef.current = true;
      setEditorStatus('Python');
      setWorkspaceError(null);
    },
    [modelId, setCodeSource, setGeneratedCode, settings.modelStoragePath]
  );

  useEffect(() => {
    let cancelled = false;
    setCurrentDirectory('');
    setWorkspaceEntries([]);
    setWorkspaceError(null);
    setWorkspaceLoading(true);
    setEditorStatus('Python');

    void (async () => {
      const defaultFile = 'huggingbox_main.py';
      try {
        let content: string;
        try {
          content = await readModelWorkspaceFile(modelId, settings.modelStoragePath, defaultFile);
        } catch {
          content = initialCodeRef.current || '';
          await writeModelWorkspaceFile(modelId, settings.modelStoragePath, defaultFile, content);
        }

        if (cancelled) return;
        setSelectedFilePath(defaultFile);
        setEditorCode(content);
        setGeneratedCode(content);
        setCodeSource('edited');
        skipNextAutosaveRef.current = true;

        await refreshWorkspaceEntries('');
      } catch (error) {
        if (!cancelled) {
          setWorkspaceError(`Workspace init failed: ${String(error)}`);
        }
      } finally {
        if (!cancelled) {
          setWorkspaceLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [modelId, refreshWorkspaceEntries, setCodeSource, setGeneratedCode, settings.modelStoragePath]);

  useEffect(() => {
    if (!selectedFilePath) return undefined;
    if (skipNextAutosaveRef.current) {
      skipNextAutosaveRef.current = false;
      return undefined;
    }

    setEditorStatus('Saving...');
    const timer = setTimeout(() => {
      void (async () => {
        try {
          await writeModelWorkspaceFile(
            modelId,
            settings.modelStoragePath,
            selectedFilePath,
            editorCode
          );
          setEditorStatus('Saved');
          setTimeout(() => setEditorStatus('Python'), 1000);
        } catch (error) {
          setEditorStatus('Save failed');
          setWorkspaceError(`Failed saving ${selectedFilePath}: ${String(error)}`);
        }
      })();
    }, 450);

    return () => clearTimeout(timer);
  }, [editorCode, modelId, selectedFilePath, settings.modelStoragePath]);

  const handleCodeChange = useCallback(
    (nextCode: string) => {
      setEditorCode(nextCode);
      setCodeSource('edited');
    },
    [setCodeSource]
  );

  const handleOpenDirectory = useCallback(
    async (relativePath: string) => {
      setCurrentDirectory(relativePath);
      setWorkspaceLoading(true);
      try {
        await refreshWorkspaceEntries(relativePath);
      } catch (error) {
        setWorkspaceError(`Could not open folder: ${String(error)}`);
      } finally {
        setWorkspaceLoading(false);
      }
    },
    [refreshWorkspaceEntries]
  );

  const handleNavigateUp = useCallback(async () => {
    const next = parentDirectory(currentDirectory);
    setCurrentDirectory(next);
    setWorkspaceLoading(true);
    try {
      await refreshWorkspaceEntries(next);
    } catch (error) {
      setWorkspaceError(`Could not load folder: ${String(error)}`);
    } finally {
      setWorkspaceLoading(false);
    }
  }, [currentDirectory, refreshWorkspaceEntries]);

  const handleRefresh = useCallback(async () => {
    setWorkspaceLoading(true);
    try {
      await refreshWorkspaceEntries(currentDirectory);
    } catch (error) {
      setWorkspaceError(`Refresh failed: ${String(error)}`);
    } finally {
      setWorkspaceLoading(false);
    }
  }, [currentDirectory, refreshWorkspaceEntries]);

  const handleCreateFile = useCallback(async () => {
    const defaultPath = currentDirectory ? `${currentDirectory}/new_file.py` : 'new_file.py';
    const typed = window.prompt('New file path (relative to model folder):', defaultPath);
    if (typed === null) return;
    const relativePath = normalizeRelativePath(typed);
    if (!relativePath) return;
    try {
      await createModelWorkspaceFile(modelId, settings.modelStoragePath, relativePath);
      const dir = parentDirectory(relativePath);
      setCurrentDirectory(dir);
      await refreshWorkspaceEntries(dir);
      await openWorkspaceFile(relativePath);
    } catch (error) {
      setWorkspaceError(`Could not create file: ${String(error)}`);
    }
  }, [currentDirectory, modelId, openWorkspaceFile, refreshWorkspaceEntries, settings.modelStoragePath]);

  const handleCreateFolder = useCallback(async () => {
    const defaultPath = currentDirectory ? `${currentDirectory}/new_folder` : 'new_folder';
    const typed = window.prompt('New folder path (relative to model folder):', defaultPath);
    if (typed === null) return;
    const relativePath = normalizeRelativePath(typed);
    if (!relativePath) return;
    try {
      await createModelWorkspaceDirectory(modelId, settings.modelStoragePath, relativePath);
      const dir = parentDirectory(relativePath);
      setCurrentDirectory(dir);
      await refreshWorkspaceEntries(dir);
    } catch (error) {
      setWorkspaceError(`Could not create folder: ${String(error)}`);
    }
  }, [currentDirectory, modelId, refreshWorkspaceEntries, settings.modelStoragePath]);

  const saveEditorFile = useCallback(async () => {
    const targetPath = selectedFilePath ?? 'huggingbox_main.py';
    setEditorStatus('Saving...');
    await writeModelWorkspaceFile(
      modelId,
      settings.modelStoragePath,
      targetPath,
      editorCode
    );
    setEditorStatus('Saved');
    setTimeout(() => setEditorStatus('Python'), 1000);
    return targetPath;
  }, [editorCode, modelId, selectedFilePath, settings.modelStoragePath]);

  async function handleRun() {
    const trimmedInput = inputValue.trim();
    if (supportsMultimodalModes) {
      if (multimodalTask === 'visual-question-answering') {
        if (!multimodalImagePath.trim()) {
          await messageDialog('Select an image before running visual question answering.', {
            kind: 'warning',
          });
          return;
        }
        if (!trimmedInput) {
          await messageDialog('Enter a question before running visual question answering.', {
            kind: 'warning',
          });
          return;
        }
      }
      if (multimodalTask === 'image-captioning' && !multimodalImagePath.trim()) {
        await messageDialog('Select an image before running image captioning.', {
          kind: 'warning',
        });
        return;
      }
      if (multimodalTask === 'document-understanding' && !multimodalDocumentPath.trim()) {
        await messageDialog('Select a document image or PDF before running document understanding.', {
          kind: 'warning',
        });
        return;
      }
    }
    if (supportsDiffusionModes) {
      if (!trimmedInput) {
        await messageDialog('Enter a prompt before running this diffusion model.', {
          kind: 'warning',
        });
        return;
      }
      if ((diffusionMode === 'image-to-image' || diffusionMode === 'inpainting') && !sourceImagePath.trim()) {
        await messageDialog('Select a source image before running this diffusion mode.', {
          kind: 'warning',
        });
        return;
      }
      if (diffusionMode === 'inpainting' && !maskImagePath.trim()) {
        await messageDialog('Select a mask image before running inpainting.', {
          kind: 'warning',
        });
        return;
      }
    }
    if (
      ['automatic-speech-recognition', 'audio-classification'].includes(model.pipeline_tag ?? '') &&
      !trimmedInput
    ) {
      await messageDialog('Select an audio file before running this model.', {
        kind: 'warning',
      });
      return;
    }

    try {
      const scriptRelativePath = await saveEditorFile();
      runCode({
        modelId,
        storagePath: settings.modelStoragePath,
        hfToken: settings.hfToken,
        pipelineTag: model.pipeline_tag,
        preferredDevice: settings.preferredDevice,
        selectedGpuId: settings.selectedGpuId,
        userInput: trimmedInput || undefined,
        envStoragePath: settings.envStoragePath || undefined,
        scriptRelativePath,
        runtimeSourceCode: editorCode,
        runMode,
        diffusionMode: supportsDiffusionModes ? diffusionMode : undefined,
        outputDir: supportsDiffusionModes ? 'outputs' : undefined,
        negativePrompt: supportsDiffusionModes ? negativePrompt : undefined,
        steps: supportsDiffusionModes ? steps : undefined,
        guidanceScale: supportsDiffusionModes ? guidanceScale : undefined,
        seed: supportsDiffusionModes ? seed : undefined,
        numImages: supportsDiffusionModes ? numImages : undefined,
        strength: supportsDiffusionModes ? strength : undefined,
        sourceImagePath: supportsDiffusionModes ? sourceImagePath : undefined,
        maskImagePath: supportsDiffusionModes ? maskImagePath : undefined,
        multimodalTask: supportsMultimodalModes ? multimodalTask : undefined,
        imagePath: supportsMultimodalModes ? multimodalImagePath : undefined,
        documentPath: supportsMultimodalModes ? multimodalDocumentPath : undefined,
      });
    } catch (error) {
      setEditorStatus('Save failed');
      setWorkspaceError(`Failed saving before run: ${String(error)}`);
    }
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Left: Input Panel — 280px fixed */}
      <div style={{ width: '280px', minWidth: '240px', flexShrink: 0 }}>
        <InputPanel
          pipelineTag={model.pipeline_tag}
          inputValue={inputValue}
          onInputChange={setInputValue}
          runMode={runMode}
          onRunModeChange={setRunMode}
          supportsMultimodalModes={supportsMultimodalModes}
          multimodalTask={multimodalTask}
          onMultimodalTaskChange={setMultimodalTask}
          multimodalImagePath={multimodalImagePath}
          onMultimodalImagePathChange={setMultimodalImagePath}
          multimodalDocumentPath={multimodalDocumentPath}
          onMultimodalDocumentPathChange={setMultimodalDocumentPath}
          supportsDiffusionModes={supportsDiffusionModes}
          diffusionMode={diffusionMode}
          onDiffusionModeChange={setDiffusionMode}
          sourceImagePath={sourceImagePath}
          onSourceImagePathChange={setSourceImagePath}
          maskImagePath={maskImagePath}
          onMaskImagePathChange={setMaskImagePath}
          negativePrompt={negativePrompt}
          onNegativePromptChange={setNegativePrompt}
          steps={steps}
          onStepsChange={setSteps}
          guidanceScale={guidanceScale}
          onGuidanceScaleChange={setGuidanceScale}
          seed={seed}
          onSeedChange={setSeed}
          numImages={numImages}
          onNumImagesChange={setNumImages}
          strength={strength}
          onStrengthChange={setStrength}
          onRun={handleRun}
          onCancel={cancelExecution}
          isRunning={isRunning}
          modelSize={sizeLabel}
          modelFormat={formatLabel}
          ramEstimate={ramLabel}
        />
      </div>

      {/* Right: Analysis + Code editor + Output */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {codeGenerationError && (
          <div
            style={{
              flex: '0 0 auto',
              maxHeight: '160px',
              overflowY: 'auto',
              borderBottom: '1px solid var(--border)',
              padding: 'var(--space-sm) var(--space-md)',
              backgroundColor: 'rgba(16,185,129,0.12)',
            }}
          >
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '11px',
                color: 'var(--success)',
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                marginBottom: '6px',
              }}
            >
              Runner Diagnostics
            </div>
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '12px',
                color: 'var(--success)',
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
              }}
            >
              {codeGenerationError}
            </div>
          </div>
        )}
        {claudeAnalysis && (
          <div
            style={{
              flex: '0 0 auto',
              maxHeight: '150px',
              overflowY: 'auto',
              borderBottom: '1px solid var(--border)',
              padding: 'var(--space-sm) var(--space-md)',
              backgroundColor: 'var(--bg-secondary)',
            }}
          >
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '11px',
                color: 'var(--text-muted)',
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                marginBottom: '6px',
              }}
            >
              Claude Analysis
            </div>
            <div
              style={{
                fontFamily: '"Inter", sans-serif',
                fontSize: '13px',
                color: 'var(--text-secondary)',
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
              }}
            >
              {claudeAnalysis}
            </div>
          </div>
        )}
        {workspaceError && (
          <div
            style={{
              flex: '0 0 auto',
              borderBottom: '1px solid var(--border)',
              padding: 'var(--space-xs) var(--space-md)',
              backgroundColor: 'rgba(245,158,11,0.10)',
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: '11px',
              color: 'var(--warning)',
              whiteSpace: 'pre-wrap',
            }}
          >
            {workspaceError}
          </div>
        )}
        <div style={{ flex: '0 0 55%', overflow: 'hidden', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', height: '100%' }}>
            <FileExplorer
              currentDirectory={currentDirectory}
              entries={workspaceEntries}
              loading={workspaceLoading}
              selectedFilePath={selectedFilePath}
              error={workspaceError}
              onOpenDirectory={handleOpenDirectory}
              onSelectFile={(path) => {
                void openWorkspaceFile(path);
              }}
              onRefresh={() => {
                void handleRefresh();
              }}
              onCreateFile={() => {
                void handleCreateFile();
              }}
              onCreateFolder={() => {
                void handleCreateFolder();
              }}
              onNavigateUp={() => {
                void handleNavigateUp();
              }}
            />
            <CodeEditor
              code={editorCode}
              fileName={fileName(selectedFilePath)}
              statusText={editorStatus}
              onCodeChange={handleCodeChange}
            />
          </div>
        </div>
        <div style={{ flex: '1 1 auto', overflow: 'hidden' }}>
          <OutputPanel
            modelId={modelId}
            pipelineTag={model.pipeline_tag}
            inputValue={inputValue}
            multimodalTask={supportsMultimodalModes ? multimodalTask : undefined}
            multimodalImagePath={supportsMultimodalModes ? multimodalImagePath : undefined}
            multimodalDocumentPath={supportsMultimodalModes ? multimodalDocumentPath : undefined}
          />
        </div>
      </div>
    </div>
  );
}
