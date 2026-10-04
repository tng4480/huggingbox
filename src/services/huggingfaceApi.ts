import type { HFModel, HFModelDetail } from '../stores/appStore';

const HF_API_BASE = 'https://huggingface.co/api';
const PAGE_SIZE = 24;

export interface FetchModelsParams {
  search?: string;
  pipeline_tag?: string;
  page?: number;
  limit?: number;
  sort?: 'trendingScore' | 'downloads' | 'likes' | 'lastModified';
}

export type RecommendedRuntime =
  | 'llama_cpp'
  | 'onnxruntime'
  | 'diffusers'
  | 'transformers_llm'
  | 'transformers_multimodal'
  | 'transformers_audio'
  | 'transformers_generic';

export interface ModelFormatInfo {
  hasGguf: boolean;
  hasOnnx: boolean;
  hasSafetensors: boolean;
  hasPytorch: boolean;
  recommendedRuntime: RecommendedRuntime;
  recommendationReason: string | null;
}

export async function fetchModels(
  params: FetchModelsParams = {},
  hfToken?: string,
  signal?: AbortSignal
): Promise<HFModel[]> {
  const { search, pipeline_tag, page = 0, limit = PAGE_SIZE, sort = 'trendingScore' } = params;

  const query = new URLSearchParams();
  if (search) query.set('search', search);
  if (pipeline_tag) query.set('pipeline_tag', pipeline_tag);
  query.set('limit', String(limit));
  query.set('offset', String(page * limit));
  query.set('sort', sort);
  query.set('direction', '-1');
  // Request full metadata
  query.set('full', 'true');
  query.set('config', 'true');

  const headers: HeadersInit = {};
  if (hfToken) headers['Authorization'] = `Bearer ${hfToken}`;

  const res = await fetch(`${HF_API_BASE}/models?${query}`, { headers, signal });
  if (!res.ok) throw new Error(`HF API error: ${res.status}`);

  const data: HFModel[] = await res.json();
  return data;
}

export async function fetchModelDetail(
  modelId: string,
  hfToken?: string
): Promise<HFModelDetail> {
  const headers: HeadersInit = {};
  if (hfToken) headers['Authorization'] = `Bearer ${hfToken}`;

  const query = new URLSearchParams();
  query.set('blobs', 'true');
  query.set('full', 'true');
  const res = await fetch(`${HF_API_BASE}/models/${modelId}?${query}`, { headers });
  if (!res.ok) throw new Error(`HF API error: ${res.status}`);

  const data: HFModelDetail = await res.json();
  return data;
}

type Sibling = NonNullable<HFModel['siblings']>[number];

function siblingSize(f: Sibling): number {
  const direct = typeof f.size === 'number' ? f.size : 0;
  const lfs = typeof f.lfs?.size === 'number' ? f.lfs.size : 0;
  return direct > 0 ? direct : lfs > 0 ? lfs : 0;
}

// ─── GGUF (quantization) helpers ─────────────────────────────────────────────

export interface GgufFileInfo {
  name: string;
  /** Size in bytes, or 0 when unknown (browse list API does not return sizes). */
  size: number;
  /** Quantization label parsed from the file name, e.g. "Q4_K_M". */
  quant: string;
}

const QUANT_RE = /(IQ\d_[A-Z0-9]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16|F32)/i;

// Preferred default quantization, best balance of quality and size first.
const DEFAULT_QUANT_ORDER = ['Q4_K_M', 'Q4_K_S', 'Q4_0', 'IQ4_XS', 'Q5_K_M', 'Q5_K_S', 'Q3_K_M', 'Q8_0'];

// Typical GGUF runtime overhead on top of the file size (KV cache, buffers).
export const GGUF_RAM_MULTIPLIER = 1.2;
// ~Q4_K_M bytes per parameter, used only when real file sizes are unavailable.
const GGUF_BYTES_PER_PARAM = 0.65;

/** All loadable GGUF files in the repo, smallest first. Projector (mmproj) and imatrix calibration files are not models and are excluded. */
export function getGgufFiles(model: HFModel): GgufFileInfo[] {
  return (model.siblings ?? [])
    .filter((f) => f.rfilename.toLowerCase().endsWith('.gguf') && !/mmproj|imatrix/i.test(f.rfilename))
    .map((f) => ({
      name: f.rfilename,
      size: siblingSize(f),
      quant: (f.rfilename.match(QUANT_RE)?.[1] ?? 'GGUF').toUpperCase(),
    }))
    .sort((a, b) => a.size - b.size || a.name.localeCompare(b.name));
}

/** The GGUF file that will be used: the user's pick if valid, otherwise a sensible default. */
export function resolveGgufFile(model: HFModel, selected?: string | null): GgufFileInfo | null {
  const files = getGgufFiles(model);
  if (files.length === 0) return null;
  const chosen = selected ? files.find((f) => f.name === selected) : undefined;
  if (chosen) return chosen;
  for (const quant of DEFAULT_QUANT_ORDER) {
    const hit = files.find((f) => f.quant === quant);
    if (hit) return hit;
  }
  return files[Math.floor(files.length / 2)];
}

export function hasGgufFiles(model: HFModel): boolean {
  return getGgufFiles(model).length > 0;
}

/** True when the size is a name-based guess rather than real file sizes from Hugging Face. */
export function isModelSizeEstimated(model: HFModel): boolean {
  const ggufs = getGgufFiles(model);
  if (ggufs.length > 0) return ggufs.every((f) => f.size === 0);
  if ((model.siblings ?? []).some((f) => siblingSize(f) > 0)) return false;
  const safetensorsTotal = (model as HFModelDetail).safetensors?.total;
  return !(typeof safetensorsTotal === 'number' && safetensorsTotal > 0);
}

/**
 * Estimate the size in bytes of what will actually be loaded.
 * GGUF repos hold many alternative quantizations of one model, so only the
 * selected file counts, never the sum of all of them.
 */
export function estimateModelSize(model: HFModel, selectedGguf?: string | null): number {
  const ggufs = getGgufFiles(model);
  if (ggufs.length > 0 && ggufs.some((f) => f.size > 0)) {
    return resolveGgufFile(model, selectedGguf)?.size ?? 0;
  }

  if (ggufs.length === 0) {
    const fromSiblings = (model.siblings ?? []).reduce((acc, f) => acc + siblingSize(f), 0);
    if (fromSiblings > 0) return fromSiblings;

    const safetensorsTotal = (model as HFModelDetail).safetensors?.total;
    if (typeof safetensorsTotal === 'number' && safetensorsTotal > 0) {
      return safetensorsTotal;
    }
  }

  // No real sizes available: guess from the parameter count in the name.
  const idLike = (model.modelId || model.id || '').toLowerCase();
  const match = idLike.match(/(\d+(?:\.\d+)?)\s*([bm])(?![a-z])/i);
  if (!match) return 0;
  const count = Number(match[1]);
  if (!Number.isFinite(count) || count <= 0) return 0;
  const scale = match[2].toLowerCase() === 'b' ? 1_000_000_000 : 1_000_000;
  const params = count * scale;
  // GGUF repos are quantized (~4-bit); everything else assumes fp16 (~2 bytes/parameter).
  return Math.round(params * (ggufs.length > 0 ? GGUF_BYTES_PER_PARAM : 2));
}
function getSiblingFilenames(model: HFModel): string[] {
  return (model.siblings ?? []).map((item) => item.rfilename.toLowerCase());
}

function isLlmLikePipeline(tag: string): boolean {
  return (
    tag.includes('text-generation') ||
    tag.includes('conversational') ||
    tag.includes('chat')
  );
}

function isOnnxSupportedPipeline(tag: string): boolean {
  return [
    'text-classification',
    'token-classification',
    'feature-extraction',
    'question-answering',
    'image-classification',
    'object-detection',
    'zero-shot-image-classification',
    'sentence-similarity',
  ].includes(tag);
}

function isDiffusionPipeline(tag: string): boolean {
  return ['text-to-image', 'image-to-image', 'inpainting'].includes(tag);
}

function isMultimodalPipeline(tag: string): boolean {
  return [
    'image-text-to-text',
    'image-to-text',
    'visual-question-answering',
    'document-question-answering',
  ].includes(tag);
}

function isAudioPipeline(tag: string): boolean {
  return [
    'automatic-speech-recognition',
    'audio-classification',
    'text-to-speech',
    'text-to-audio',
  ].includes(tag);
}

export function getModelFormatInfo(model: HFModel): ModelFormatInfo {
  const pipeline = (model.pipeline_tag ?? '').toLowerCase();
  const files = getSiblingFilenames(model);
  const hasGguf = files.some((name) => name.endsWith('.gguf'));
  const hasOnnx = files.some((name) => name.endsWith('.onnx') || name.endsWith('.ort'));
  const hasSafetensors = files.some((name) => name.endsWith('.safetensors'));
  const hasPytorch = files.some((name) => name.endsWith('.bin') || name.endsWith('.pt') || name.endsWith('.pth'));

  if (hasGguf && isLlmLikePipeline(pipeline)) {
    return {
      hasGguf,
      hasOnnx,
      hasSafetensors,
      hasPytorch,
      recommendedRuntime: 'llama_cpp',
      recommendationReason:
        'This repo includes GGUF weights, which are usually the best local option for LLM inference on your hardware.',
    };
  }

  if (hasOnnx && isOnnxSupportedPipeline(pipeline)) {
    return {
      hasGguf,
      hasOnnx,
      hasSafetensors,
      hasPytorch,
      recommendedRuntime: 'onnxruntime',
      recommendationReason:
        'This repo includes ONNX weights and this pipeline usually runs faster with ONNX Runtime on your hardware.',
    };
  }

  if (isDiffusionPipeline(pipeline)) {
    return {
      hasGguf,
      hasOnnx,
      hasSafetensors,
      hasPytorch,
      recommendedRuntime: 'diffusers',
      recommendationReason: 'Diffusion models are best handled through the diffusers runtime.',
    };
  }

  if (isMultimodalPipeline(pipeline)) {
    return {
      hasGguf,
      hasOnnx,
      hasSafetensors,
      hasPytorch,
      recommendedRuntime: 'transformers_multimodal',
      recommendationReason: 'This model looks multimodal, so the multimodal transformers runtime is the right default.',
    };
  }

  if (isAudioPipeline(pipeline)) {
    return {
      hasGguf,
      hasOnnx,
      hasSafetensors,
      hasPytorch,
      recommendedRuntime: 'transformers_audio',
      recommendationReason: 'This model uses an audio pipeline, so the audio runtime is the right default.',
    };
  }

  if (isLlmLikePipeline(pipeline)) {
    return {
      hasGguf,
      hasOnnx,
      hasSafetensors,
      hasPytorch,
      recommendedRuntime: 'transformers_llm',
      recommendationReason: 'This is a text-generation model without GGUF assets, so transformers is the default runtime.',
    };
  }

  return {
    hasGguf,
    hasOnnx,
    hasSafetensors,
    hasPytorch,
    recommendedRuntime: 'transformers_generic',
    recommendationReason: hasSafetensors || hasPytorch
      ? 'This repo ships standard transformers weights, so the generic transformers runtime is the default.'
      : null,
  };
}

// Format bytes to human-readable string
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const gb = bytes / (1024 ** 3);
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / (1024 ** 2);
  if (mb >= 1) return `${mb.toFixed(0)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

// Format download count
export function formatDownloads(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

// Map pipeline_tag to category for badge colour
export type ModelCategory = 'text' | 'vision' | 'audio' | 'generation' | 'multimodal' | 'other';

export function getModelCategory(pipeline_tag: string | null | undefined): ModelCategory {
  if (!pipeline_tag) return 'other';
  const tag = pipeline_tag.toLowerCase();
  if (
    tag.includes('text-generation') ||
    tag.includes('text-classification') ||
    tag.includes('summarization') ||
    tag.includes('translation') ||
    tag.includes('question-answering') ||
    tag.includes('fill-mask') ||
    tag.includes('token-classification') ||
    tag.includes('feature-extraction') ||
    tag.includes('sentence-similarity')
  ) return 'text';
  if (
    tag.includes('image-classification') ||
    tag.includes('object-detection') ||
    tag.includes('image-segmentation') ||
    tag.includes('depth-estimation') ||
    tag.includes('image-to-image')
  ) return 'vision';
  if (
    tag.includes('automatic-speech') ||
    tag.includes('text-to-speech') ||
    tag.includes('audio')
  ) return 'audio';
  if (
    tag.includes('text-to-image') ||
    tag.includes('unconditional')
  ) return 'generation';
  if (
    tag.includes('visual') ||
    tag.includes('image-to-text') ||
    tag.includes('document')
  ) return 'multimodal';
  return 'other';
}

export const CATEGORY_BADGE_COLORS: Record<ModelCategory, string> = {
  text: '#2563EB',
  vision: '#7C3AED',
  audio: '#059669',
  generation: '#DB2777',
  multimodal: '#D97706',
  other: '#6B7280',
};

// Pipeline filter options for the Browse view dropdown
export const PIPELINE_OPTIONS = [
  { label: 'All Pipelines', value: '' },
  { label: 'Text Generation', value: 'text-generation' },
  { label: 'Text Classification', value: 'text-classification' },
  { label: 'Summarization', value: 'summarization' },
  { label: 'Question Answering', value: 'question-answering' },
  { label: 'Image Classification', value: 'image-classification' },
  { label: 'Object Detection', value: 'object-detection' },
  { label: 'Image Segmentation', value: 'image-segmentation' },
  { label: 'Speech Recognition', value: 'automatic-speech-recognition' },
  { label: 'Text to Speech', value: 'text-to-speech' },
  { label: 'Text to Image', value: 'text-to-image' },
  { label: 'Visual QA', value: 'visual-question-answering' },
  { label: 'Feature Extraction', value: 'feature-extraction' },
];

export const SORT_OPTIONS = [
  { label: 'Trending', value: 'trendingScore' },
  { label: 'Most Downloads', value: 'downloads' },
  { label: 'Most Likes', value: 'likes' },
  { label: 'Recently Updated', value: 'lastModified' },
];

export const SIZE_OPTIONS = [
  { label: 'All Sizes', value: '' },
  { label: 'Small (< 1 GB)', value: 'small' },
  { label: 'Medium (1–5 GB)', value: 'medium' },
  { label: 'Large (5–20 GB)', value: 'large' },
  { label: 'Very Large (> 20 GB)', value: 'xlarge' },
];

const SIZE_RANGES: Record<string, [number, number]> = {
  small:  [0,           1 * 1024 ** 3],
  medium: [1 * 1024 ** 3, 5 * 1024 ** 3],
  large:  [5 * 1024 ** 3, 20 * 1024 ** 3],
  xlarge: [20 * 1024 ** 3, Infinity],
};

export function modelMatchesSizeFilter(model: HFModel, sizeFilter: string): boolean {
  if (!sizeFilter) return true;
  const range = SIZE_RANGES[sizeFilter];
  if (!range) return true;
  const size = estimateModelSize(model);
  if (size === 0) return true; // unknown size — show it
  return size >= range[0] && size < range[1];
}
