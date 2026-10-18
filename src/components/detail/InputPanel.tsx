import { useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { Play, Square, Upload } from 'lucide-react';
import Button from '../shared/Button';

export type DiffusionMode = 'text-to-image' | 'image-to-image' | 'inpainting';
export type MultimodalTask = 'visual-question-answering' | 'image-captioning' | 'document-understanding';

interface InputPanelProps {
  pipelineTag: string | null | undefined;
  inputValue: string;
  onInputChange: (val: string) => void;
  runMode: 'prepared' | 'direct';
  onRunModeChange: (mode: 'prepared' | 'direct') => void;
  onRun: () => void;
  onCancel: () => void;
  isRunning: boolean;
  disabled?: boolean;
  modelSize?: string;
  modelFormat?: string;
  ramEstimate?: string;
  supportsDiffusionModes?: boolean;
  diffusionMode?: DiffusionMode;
  onDiffusionModeChange?: (mode: DiffusionMode) => void;
  sourceImagePath?: string;
  onSourceImagePathChange?: (path: string) => void;
  maskImagePath?: string;
  onMaskImagePathChange?: (path: string) => void;
  negativePrompt?: string;
  onNegativePromptChange?: (value: string) => void;
  steps?: number;
  onStepsChange?: (value: number) => void;
  guidanceScale?: number;
  onGuidanceScaleChange?: (value: number) => void;
  seed?: string;
  onSeedChange?: (value: string) => void;
  numImages?: number;
  onNumImagesChange?: (value: number) => void;
  strength?: number;
  onStrengthChange?: (value: number) => void;
  supportsMultimodalModes?: boolean;
  multimodalTask?: MultimodalTask;
  onMultimodalTaskChange?: (task: MultimodalTask) => void;
  multimodalImagePath?: string;
  onMultimodalImagePathChange?: (path: string) => void;
  multimodalDocumentPath?: string;
  onMultimodalDocumentPathChange?: (path: string) => void;
}

function getInputType(pipeline: string | null | undefined): 'text' | 'image' | 'audio' | 'prompt' {
  if (!pipeline) return 'text';
  if (
    [
      'image-classification',
      'image-segmentation',
      'object-detection',
      'depth-estimation',
      'image-to-image',
      'image-text-to-text',
      'image-to-text',
      'inpainting',
    ].includes(pipeline)
  ) {
    return 'image';
  }
  if (['automatic-speech-recognition', 'audio-classification'].includes(pipeline)) return 'audio';
  if (pipeline === 'text-to-image') return 'prompt';
  return 'text';
}

function getInputLabel(pipeline: string | null | undefined): string {
  if (!pipeline) return 'Input';
  const labels: Record<string, string> = {
    'text-generation': 'Prompt',
    'text2text-generation': 'Input Text',
    summarization: 'Text to Summarize',
    'question-answering': 'Question + Context',
    translation_en_to_fr: 'English Text',
    'fill-mask': 'Text with [MASK]',
    'text-classification': 'Input Text',
    'token-classification': 'Input Text',
    'feature-extraction': 'Input Text',
    'text-to-image': 'Prompt',
    'image-to-image': 'Image',
    inpainting: 'Image',
    'image-text-to-text': 'Image',
    'image-to-text': 'Image',
    'image-classification': 'Image',
    'object-detection': 'Image',
    'image-segmentation': 'Image',
    'depth-estimation': 'Image',
    'automatic-speech-recognition': 'Audio File',
    'audio-classification': 'Audio File',
    'text-to-speech': 'Text to Speak',
    'text-to-audio': 'Text to Speak',
  };
  return labels[pipeline] ?? 'Input';
}

function getPlaceholder(pipeline: string | null | undefined): string {
  if (!pipeline) return 'Enter your input here...';
  const placeholders: Record<string, string> = {
    'text-generation': 'Once upon a time...',
    'text2text-generation': 'Enter text to transform...',
    summarization: 'Paste a long article or document here...',
    'question-answering': 'Question: What is the capital of France?\nContext: Paris is the capital...',
    'fill-mask': 'The capital of France is [MASK].',
    'text-classification': 'Enter text to classify...',
    'token-classification': 'Enter text for entity recognition...',
    'feature-extraction': 'Enter text to embed...',
    'text-to-image': 'A photorealistic image of a mountain at sunset...',
    'text-to-speech': 'Hello, this text will be spoken aloud.',
    'text-to-audio': 'Hello, this text will be spoken aloud.',
  };
  return placeholders[pipeline] ?? 'Enter your input here...';
}

function isImageDataUrl(value: string): boolean {
  return value.startsWith('data:image/');
}

function clampInteger(value: number, fallback: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}

function clampFloat(value: number, fallback: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function useLocalImagePreview(path: string): string | null {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let nextUrl: string | null = null;

    async function load() {
      try {
        const bytes = await invoke<number[]>('read_binary_file', { filePath: path });
        if (cancelled) return;
        const lower = path.toLowerCase();
        const type = lower.endsWith('.png')
          ? 'image/png'
          : lower.endsWith('.webp')
            ? 'image/webp'
            : lower.endsWith('.gif')
              ? 'image/gif'
              : 'image/jpeg';
        nextUrl = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type }));
        setPreviewUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return nextUrl;
        });
      } catch {
        if (!cancelled) {
          setPreviewUrl((previous) => {
            if (previous) URL.revokeObjectURL(previous);
            return null;
          });
        }
      }
    }

    if (path.trim()) {
      void load();
    } else {
      setPreviewUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return null;
      });
    }

    return () => {
      cancelled = true;
      if (nextUrl) {
        URL.revokeObjectURL(nextUrl);
      }
    };
  }, [path]);

  return previewUrl;
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span
        style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '11px',
          color: 'var(--text-muted)',
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '11px',
          color: 'var(--text-secondary)',
        }}
      >
        {value}
      </span>
    </div>
  );
}

function UploadCard({
  title,
  subtitle,
  fileLabel,
  previewUrl,
  onClick,
  onDropPath,
}: {
  title: string;
  subtitle: string;
  fileLabel?: string;
  previewUrl?: string | null;
  onClick: () => void;
  onDropPath?: (path: string) => void;
}) {
  return (
    <div
      onClick={onClick}
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={(e) => {
        e.preventDefault();
        const candidate = e.dataTransfer.files?.[0] as File & { path?: string };
        if (candidate?.path && onDropPath) {
          onDropPath(candidate.path);
        }
      }}
      style={{
        minHeight: '120px',
        border: '1px dashed var(--border)',
        borderRadius: '4px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 'var(--space-sm)',
        cursor: 'pointer',
        color: 'var(--text-muted)',
        transition: 'border-color 150ms, color 150ms',
        padding: 'var(--space-sm)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'var(--accent-primary)';
        e.currentTarget.style.color = 'var(--text-secondary)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border)';
        e.currentTarget.style.color = 'var(--text-muted)';
      }}
    >
      <Upload size={24} strokeWidth={1.5} />
      <span style={{ fontFamily: '"Inter", sans-serif', fontSize: '13px', textAlign: 'center' }}>
        {title}
      </span>
      <span style={{ fontFamily: '"Inter", sans-serif', fontSize: '11px', textAlign: 'center' }}>
        {subtitle}
      </span>
      {fileLabel && (
        <span
          style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: '11px',
            color: 'var(--text-secondary)',
            textAlign: 'center',
            wordBreak: 'break-all',
          }}
        >
          {fileLabel}
        </span>
      )}
      {previewUrl && (
        <img
          src={previewUrl}
          alt={title}
          style={{ maxWidth: '100%', maxHeight: '130px', borderRadius: '4px', objectFit: 'contain' }}
        />
      )}
    </div>
  );
}

export default function InputPanel({
  pipelineTag,
  inputValue,
  onInputChange,
  runMode,
  onRunModeChange,
  onRun,
  onCancel,
  isRunning,
  disabled = false,
  modelSize,
  modelFormat,
  ramEstimate,
  supportsDiffusionModes = false,
  diffusionMode = 'text-to-image',
  onDiffusionModeChange,
  sourceImagePath = '',
  onSourceImagePathChange,
  maskImagePath = '',
  onMaskImagePathChange,
  negativePrompt = '',
  onNegativePromptChange,
  steps = 30,
  onStepsChange,
  guidanceScale = 7.5,
  onGuidanceScaleChange,
  seed = '',
  onSeedChange,
  numImages = 1,
  onNumImagesChange,
  strength = 0.75,
  onStrengthChange,
  supportsMultimodalModes = false,
  multimodalTask = 'visual-question-answering',
  onMultimodalTaskChange,
  multimodalImagePath = '',
  onMultimodalImagePathChange,
  multimodalDocumentPath = '',
  onMultimodalDocumentPathChange,
}: InputPanelProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const inputType = getInputType(pipelineTag);
  const [batchMode, setBatchMode] = useState(false);
  const [qaContext, setQaContext] = useState('');
  const [qaQuestion, setQaQuestion] = useState('');
  const isQa = pipelineTag === 'question-answering';
  const isTts = pipelineTag === 'text-to-speech' || pipelineTag === 'text-to-audio';
  const supportsBatch = useMemo(
    () => ['text-classification', 'summarization', 'feature-extraction'].includes(pipelineTag ?? ''),
    [pipelineTag]
  );
  const isDiffusion = supportsDiffusionModes;
  const isMultimodal = supportsMultimodalModes;
  const sourcePreviewUrl = useLocalImagePreview(sourceImagePath);
  const maskPreviewUrl = useLocalImagePreview(maskImagePath);
  const multimodalImagePreviewUrl = useLocalImagePreview(multimodalImagePath);
  const multimodalDocumentPreviewUrl = useLocalImagePreview(
    multimodalDocumentPath.toLowerCase().endsWith('.pdf') ? '' : multimodalDocumentPath
  );
  const displayInputValue = useMemo(() => {
    if (!inputValue.startsWith('__HBJSON__:')) return inputValue;
    const raw = inputValue.slice('__HBJSON__:'.length);
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed.join('\n');
    } catch {
      // ignore
    }
    return '';
  }, [inputValue]);
  const imagePreview = useMemo(() => {
    if (inputValue.startsWith('__HBIMG__:')) return inputValue.slice('__HBIMG__:'.length);
    if (isImageDataUrl(inputValue)) return inputValue;
    return '';
  }, [inputValue]);
  const audioFileLabel = useMemo(() => {
    if (inputType !== 'audio' || !inputValue) return '';
    const normalized = inputValue.replace(/\\/g, '/');
    return normalized.split('/').pop() ?? normalized;
  }, [inputType, inputValue]);
  const sourceImageLabel = useMemo(() => {
    const normalized = sourceImagePath.replace(/\\/g, '/');
    return normalized ? normalized.split('/').pop() ?? normalized : '';
  }, [sourceImagePath]);
  const maskImageLabel = useMemo(() => {
    const normalized = maskImagePath.replace(/\\/g, '/');
    return normalized ? normalized.split('/').pop() ?? normalized : '';
  }, [maskImagePath]);
  const multimodalImageLabel = useMemo(() => {
    const normalized = multimodalImagePath.replace(/\\/g, '/');
    return normalized ? normalized.split('/').pop() ?? normalized : '';
  }, [multimodalImagePath]);
  const multimodalDocumentLabel = useMemo(() => {
    const normalized = multimodalDocumentPath.replace(/\\/g, '/');
    return normalized ? normalized.split('/').pop() ?? normalized : '';
  }, [multimodalDocumentPath]);

  useEffect(() => {
    if (!isQa || !inputValue.startsWith('__HBJSON__:')) return;
    try {
      const parsed = JSON.parse(inputValue.slice('__HBJSON__:'.length));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        setQaQuestion(typeof parsed.question === 'string' ? parsed.question : '');
        setQaContext(typeof parsed.context === 'string' ? parsed.context : '');
      }
    } catch {
      // ignore malformed input cache
    }
  }, [inputValue, isQa]);

  async function handleImageFile(file: File | undefined) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      if (result.startsWith('data:image/')) {
        onInputChange(`__HBIMG__:${result}`);
      }
    };
    reader.readAsDataURL(file);
  }

  async function chooseFile(extensions: string[]): Promise<string> {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: 'Files', extensions }],
    });
    return typeof selected === 'string' ? selected : '';
  }

  async function handleChooseAudioFile() {
    const selected = await chooseFile(['wav', 'mp3', 'flac']);
    if (selected.trim()) {
      onInputChange(selected);
    }
  }

  async function handleChooseDiffusionImage(kind: 'source' | 'mask') {
    const selected = await chooseFile(['png', 'jpg', 'jpeg', 'webp']);
    if (!selected.trim()) return;
    if (kind === 'source') {
      onSourceImagePathChange?.(selected);
    } else {
      onMaskImagePathChange?.(selected);
    }
  }

  async function handleChooseMultimodalFile(kind: 'image' | 'document') {
    const selected = await chooseFile(
      kind === 'image' ? ['png', 'jpg', 'jpeg', 'webp'] : ['png', 'jpg', 'jpeg', 'webp', 'pdf']
    );
    if (!selected.trim()) return;
    if (kind === 'image') {
      onMultimodalImagePathChange?.(selected);
    } else {
      onMultimodalDocumentPathChange?.(selected);
    }
  }

  function emitTextInput(val: string) {
    if (supportsBatch && batchMode) {
      const items = val
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
      onInputChange(`__HBJSON__:${JSON.stringify(items)}`);
      return;
    }
    onInputChange(val);
  }

  function emitQaInput(nextQuestion: string, nextContext: string) {
    onInputChange(`__HBJSON__:${JSON.stringify({ question: nextQuestion, context: nextContext })}`);
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
        backgroundColor: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border)',
      }}
    >
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: 'var(--space-lg)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-sm)',
        }}
      >
        <span
          style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: '11px',
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}
        >
          {isMultimodal ? 'Multimodal Input' : isDiffusion ? 'Prompt' : getInputLabel(pipelineTag)}
        </span>

        {isMultimodal ? (
          <>
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                padding: '10px',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                backgroundColor: 'var(--bg-primary)',
              }}
            >
              <span style={sectionLabelStyle}>Multimodal Task</span>
              {(
                [
                  ['visual-question-answering', 'Visual Q&A'],
                  ['image-captioning', 'Image Captioning'],
                  ['document-understanding', 'Document Understanding'],
                ] as Array<[MultimodalTask, string]>
              ).map(([task, label]) => (
                <label
                  key={task}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontFamily: '"Inter", sans-serif',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    opacity: isRunning ? 0.6 : 1,
                  }}
                >
                  <input
                    type="radio"
                    name="multimodal-task"
                    checked={multimodalTask === task}
                    onChange={() => onMultimodalTaskChange?.(task)}
                    disabled={isRunning}
                    style={{ accentColor: 'var(--accent-primary)' }}
                  />
                  {label}
                </label>
              ))}
            </div>

            {multimodalTask !== 'document-understanding' && (
              <UploadCard
                title={multimodalImageLabel ? 'Click to replace image' : 'Click to upload image'}
                subtitle="PNG, JPG, WebP"
                fileLabel={multimodalImageLabel}
                previewUrl={multimodalImagePreviewUrl}
                onClick={() => {
                  void handleChooseMultimodalFile('image');
                }}
                onDropPath={(path) => onMultimodalImagePathChange?.(path)}
              />
            )}

            {multimodalTask === 'document-understanding' && (
              <UploadCard
                title={multimodalDocumentLabel ? 'Click to replace document' : 'Click to upload document'}
                subtitle="PNG, JPG, WebP, PDF"
                fileLabel={multimodalDocumentLabel}
                previewUrl={multimodalDocumentPreviewUrl}
                onClick={() => {
                  void handleChooseMultimodalFile('document');
                }}
                onDropPath={(path) => onMultimodalDocumentPathChange?.(path)}
              />
            )}

            {multimodalTask === 'visual-question-answering' && (
              <textarea
                value={inputValue}
                onChange={(e) => onInputChange(e.target.value)}
                placeholder="Ask a question about the uploaded image..."
                style={{
                  minHeight: '110px',
                  resize: 'vertical',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  padding: 'var(--space-sm)',
                  fontFamily: '"Inter", sans-serif',
                  fontSize: '13px',
                  color: 'var(--text-primary)',
                  outline: 'none',
                  lineHeight: 1.5,
                }}
              />
            )}

            {multimodalTask === 'document-understanding' && (
              <textarea
                value={inputValue}
                onChange={(e) => onInputChange(e.target.value)}
                placeholder="Extract the important text and structure from this document."
                style={{
                  minHeight: '110px',
                  resize: 'vertical',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  padding: 'var(--space-sm)',
                  fontFamily: '"Inter", sans-serif',
                  fontSize: '13px',
                  color: 'var(--text-primary)',
                  outline: 'none',
                  lineHeight: 1.5,
                }}
              />
            )}
          </>
        ) : isDiffusion ? (
          <>
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                padding: '10px',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                backgroundColor: 'var(--bg-primary)',
              }}
            >
              <span
                style={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                }}
              >
                Diffusion Mode
              </span>
              {(['text-to-image', 'image-to-image', 'inpainting'] as DiffusionMode[]).map((mode) => (
                <label
                  key={mode}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontFamily: '"Inter", sans-serif',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    opacity: isRunning ? 0.6 : 1,
                  }}
                >
                  <input
                    type="radio"
                    name="diffusion-mode"
                    checked={diffusionMode === mode}
                    onChange={() => onDiffusionModeChange?.(mode)}
                    disabled={isRunning}
                    style={{ accentColor: 'var(--accent-primary)' }}
                  />
                  {mode === 'text-to-image' ? 'Text to Image' : mode === 'image-to-image' ? 'Image to Image' : 'Inpainting'}
                </label>
              ))}
            </div>

            <textarea
              value={inputValue}
              onChange={(e) => onInputChange(e.target.value)}
              placeholder="A cinematic portrait of a robot in rainy neon streets..."
              style={{
                minHeight: '120px',
                resize: 'vertical',
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: 'var(--space-sm)',
                fontFamily: '"Inter", sans-serif',
                fontSize: '13px',
                color: 'var(--text-primary)',
                outline: 'none',
                lineHeight: 1.5,
              }}
            />

            {(diffusionMode === 'image-to-image' || diffusionMode === 'inpainting') && (
              <>
                <span
                  style={{
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    marginTop: '4px',
                  }}
                >
                  Source Image
                </span>
                <UploadCard
                  title={sourceImageLabel ? 'Click to replace source image' : 'Click to upload source image'}
                  subtitle="PNG, JPG, WebP"
                  fileLabel={sourceImageLabel}
                  previewUrl={sourcePreviewUrl}
                  onClick={() => {
                    void handleChooseDiffusionImage('source');
                  }}
                  onDropPath={(path) => onSourceImagePathChange?.(path)}
                />
              </>
            )}

            {diffusionMode === 'inpainting' && (
              <>
                <span
                  style={{
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    marginTop: '4px',
                  }}
                >
                  Mask Image
                </span>
                <UploadCard
                  title={maskImageLabel ? 'Click to replace mask image' : 'Click to upload mask image'}
                  subtitle="White reveals editable areas"
                  fileLabel={maskImageLabel}
                  previewUrl={maskPreviewUrl}
                  onClick={() => {
                    void handleChooseDiffusionImage('mask');
                  }}
                  onDropPath={(path) => onMaskImagePathChange?.(path)}
                />
              </>
            )}

            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
                marginTop: '8px',
                padding: '10px',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                backgroundColor: 'var(--bg-primary)',
              }}
            >
              <span
                style={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                }}
              >
                Generation Params
              </span>
              <LabeledNumberInput
                label="Steps"
                value={String(steps)}
                onChange={(value) => onStepsChange?.(clampInteger(Number(value), 30, 1, 200))}
              />
              <LabeledNumberInput
                label="Guidance"
                value={String(guidanceScale)}
                step="0.1"
                onChange={(value) => onGuidanceScaleChange?.(clampFloat(Number(value), 7.5, 0, 50))}
              />
              <LabeledNumberInput
                label="Images"
                value={String(numImages)}
                onChange={(value) => onNumImagesChange?.(clampInteger(Number(value), 1, 1, 8))}
              />
              {(diffusionMode === 'image-to-image' || diffusionMode === 'inpainting') && (
                <LabeledNumberInput
                  label="Strength"
                  value={String(strength)}
                  step="0.05"
                  min="0"
                  max="1"
                  onChange={(value) => onStrengthChange?.(clampFloat(Number(value), 0.75, 0, 1))}
                />
              )}
              <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={paramLabelStyle}>Negative Prompt</span>
                <textarea
                  value={negativePrompt}
                  onChange={(e) => onNegativePromptChange?.(e.target.value)}
                  placeholder="blurry, low quality, distorted"
                  style={smallTextareaStyle}
                />
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={paramLabelStyle}>Seed</span>
                <input
                  type="text"
                  value={seed}
                  onChange={(e) => onSeedChange?.(e.target.value)}
                  placeholder="Blank = random"
                  style={inputStyle}
                />
              </label>
            </div>
          </>
        ) : (
          <>
            {(inputType === 'text' || inputType === 'prompt' || isTts) && !isQa && (
              <>
                {supportsBatch && (
                  <label
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontFamily: '"Inter", sans-serif',
                      fontSize: '12px',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={batchMode}
                      onChange={(e) => {
                        setBatchMode(e.target.checked);
                        emitTextInput(displayInputValue);
                      }}
                    />
                    Batch mode (one input per line)
                  </label>
                )}
                <textarea
                  value={displayInputValue}
                  onChange={(e) => emitTextInput(e.target.value)}
                  placeholder={batchMode ? 'One input per line...' : getPlaceholder(pipelineTag)}
                  style={{
                    flex: 1,
                    minHeight: '120px',
                    resize: 'vertical',
                    backgroundColor: 'var(--bg-primary)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                    padding: 'var(--space-sm)',
                    fontFamily: '"Inter", sans-serif',
                    fontSize: '13px',
                    color: 'var(--text-primary)',
                    outline: 'none',
                    transition: 'border-color 150ms',
                    lineHeight: 1.5,
                  }}
                />
              </>
            )}

            {isQa && (
              <>
                <textarea
                  value={qaContext}
                  onChange={(e) => {
                    const next = e.target.value;
                    setQaContext(next);
                    emitQaInput(qaQuestion, next);
                  }}
                  placeholder="Context..."
                  style={{ ...smallTextareaStyle, minHeight: '120px' }}
                />
                <textarea
                  value={qaQuestion}
                  onChange={(e) => {
                    const next = e.target.value;
                    setQaQuestion(next);
                    emitQaInput(next, qaContext);
                  }}
                  placeholder="Question..."
                  style={{ ...smallTextareaStyle, minHeight: '72px' }}
                />
              </>
            )}

            {inputType === 'image' && (
              <UploadCard
                title={imagePreview ? 'Click to replace image' : 'Click to upload image'}
                subtitle="PNG, JPG, WebP"
                previewUrl={imagePreview || undefined}
                onClick={() => fileRef.current?.click()}
              />
            )}

            {inputType === 'image' && (
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => {
                  void handleImageFile(e.target.files?.[0]);
                }}
              />
            )}

            {inputType === 'audio' && (
              <UploadCard
                title={audioFileLabel ? 'Click to replace audio file' : 'Click to upload audio'}
                subtitle="WAV, MP3, FLAC"
                fileLabel={audioFileLabel}
                onClick={() => {
                  void handleChooseAudioFile();
                }}
                onDropPath={(path) => onInputChange(path)}
              />
            )}
          </>
        )}
      </div>

      <div
        style={{
          padding: 'var(--space-md) var(--space-lg)',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-sm)',
          flexShrink: 0,
        }}
      >
        <details className="execution-options"><summary>Advanced execution</summary>
          <label style={radioLabelStyle(isRunning)}>
            <input
              type="radio"
              name="run-mode"
              checked={runMode === 'prepared'}
              onChange={() => onRunModeChange('prepared')}
              disabled={isRunning}
              style={{ accentColor: 'var(--accent-primary)' }}
            />
            Prepared Run: dependency checks and compatibility probe
          </label>
          <label style={radioLabelStyle(isRunning)}>
            <input
              type="radio"
              name="run-mode"
              checked={runMode === 'direct'}
              onChange={() => onRunModeChange('direct')}
              disabled={isRunning}
              style={{ accentColor: 'var(--accent-primary)' }}
            />
            Direct Run: use a previously prepared model without repair checks
          </label>
        </details>
        {!isRunning ? (
          <Button
            variant="primary"
            icon={<Play size={13} fill="currentColor" strokeWidth={0} />}
            onClick={onRun}
            disabled={disabled}
            style={{ width: '100%', justifyContent: 'center' }}
          >
            Run
          </Button>
        ) : (
          <Button
            variant="secondary"
            icon={<Square size={13} fill="currentColor" strokeWidth={0} />}
            onClick={onCancel}
            style={{
              width: '100%',
              justifyContent: 'center',
              borderColor: 'var(--error)',
              color: 'var(--error)',
            }}
          >
            Stop
          </Button>
        )}
      </div>

      {(modelSize || modelFormat || ramEstimate) && (
        <div
          style={{
            padding: 'var(--space-md) var(--space-lg)',
            borderTop: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            flexShrink: 0,
          }}
        >
          <span style={{ ...sectionLabelStyle, marginBottom: '4px' }}>Model Info</span>
          {modelSize && <MetaRow label="Size" value={modelSize} />}
          {modelFormat && <MetaRow label="Format" value={modelFormat} />}
          {ramEstimate && <MetaRow label="RAM Est." value={ramEstimate} />}
        </div>
      )}
    </div>
  );
}

function LabeledNumberInput({
  label,
  value,
  onChange,
  step = '1',
  min,
  max,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  step?: string;
  min?: string;
  max?: string;
}) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <span style={paramLabelStyle}>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value)}
        style={inputStyle}
      />
    </label>
  );
}

const sectionLabelStyle = {
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: '11px',
  color: 'var(--text-muted)',
  textTransform: 'uppercase' as const,
  letterSpacing: '0.06em',
};

const paramLabelStyle = {
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: '11px',
  color: 'var(--text-muted)',
};

const inputStyle = {
  height: '32px',
  border: '1px solid var(--border)',
  borderRadius: '4px',
  backgroundColor: 'var(--bg-secondary)',
  color: 'var(--text-primary)',
  padding: '0 8px',
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: '12px',
  outline: 'none',
};

const smallTextareaStyle = {
  minHeight: '70px',
  resize: 'vertical' as const,
  backgroundColor: 'var(--bg-secondary)',
  border: '1px solid var(--border)',
  borderRadius: '4px',
  padding: '8px',
  fontFamily: '"Inter", sans-serif',
  fontSize: '13px',
  color: 'var(--text-primary)',
  outline: 'none',
  lineHeight: 1.5,
};

function radioLabelStyle(isRunning: boolean) {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: '"Inter", sans-serif',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    cursor: isRunning ? 'default' : 'pointer',
    opacity: isRunning ? 0.6 : 1,
  } as const;
}
