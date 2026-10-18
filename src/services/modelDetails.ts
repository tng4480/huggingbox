import { invoke } from '@tauri-apps/api/core';
import { fetchModelDetail } from './huggingfaceApi';
import type { HFModelDetail } from '../stores/appStore';

export async function loadModelDetail(id: string, storagePath: string, token?: string): Promise<HFModelDetail> {
  try {
    const cached = await invoke<HFModelDetail>('read_model_metadata', {modelId: id, storagePath});
    if (cached.id === id) return cached;
  } catch { /* New models and browser previews use the Hub. */ }
  return fetchModelDetail(id, token);
}
