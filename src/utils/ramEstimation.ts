import type { HFModel } from '../stores/appStore';
import { estimateModelSize, GGUF_RAM_MULTIPLIER, hasGgufFiles } from '../services/huggingfaceApi';

// RAM multipliers per pipeline type
const RAM_MULTIPLIERS: Record<string, number> = {
  'text-generation': 2.5,
  'text2text-generation': 2.5,
  'summarization': 2.5,
  'translation': 2.5,
  'question-answering': 2.0,
  'text-classification': 2.0,
  'token-classification': 2.0,
  'fill-mask': 2.0,
  'feature-extraction': 1.5,
  'sentence-similarity': 1.5,
  'image-classification': 1.5,
  'object-detection': 1.5,
  'image-segmentation': 1.5,
  'depth-estimation': 1.5,
  'automatic-speech-recognition': 1.5,
  'text-to-speech': 1.5,
  'audio-classification': 1.5,
  'text-to-image': 0, // flat estimate
  'image-to-image': 0,
};

const DIFFUSION_FLAT_GB = 8 * 1024 ** 3; // 8 GB flat for diffusion models

export function estimateRamBytes(model: HFModel, selectedGguf?: string | null): number {
  const tag = model.pipeline_tag ?? '';
  const sizeBytes = estimateModelSize(model, selectedGguf);

  // Diffusion models: flat estimate
  if (tag === 'text-to-image' || tag === 'image-to-image') {
    return DIFFUSION_FLAT_GB;
  }

  // GGUF weights are memory-mapped at roughly their file size plus a small
  // runtime overhead; the fp16/transformers multipliers below do not apply.
  if (hasGgufFiles(model)) {
    return sizeBytes * GGUF_RAM_MULTIPLIER;
  }

  const multiplier = RAM_MULTIPLIERS[tag] ?? 2.0;
  return sizeBytes * multiplier;
}

export type CompatibilityLevel = 'compatible' | 'tight' | 'too-large' | 'unknown';

export function getCompatibility(
  model: HFModel,
  totalRamBytes: number,
  selectedGguf?: string | null
): CompatibilityLevel {
  if (totalRamBytes === 0) return 'unknown';
  const ramEst = estimateRamBytes(model, selectedGguf);
  if (ramEst === 0) return 'unknown';
  const ratio = ramEst / totalRamBytes;
  if (ratio < 0.6) return 'compatible';
  if (ratio < 0.9) return 'tight';
  return 'too-large';
}
