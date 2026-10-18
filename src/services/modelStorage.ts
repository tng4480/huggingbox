import { invoke } from '@tauri-apps/api/core';
import type { DownloadedModel } from '../stores/appStore';

interface DownloadedModelPayload {
  id: string;
  name: string;
  pipeline_tag: string;
  size_bytes: number;
  last_used: string;
  storage_path: string;
}

export async function listDownloadedModels(storagePath: string): Promise<DownloadedModel[]> {
  const rows = await invoke<DownloadedModelPayload[]>('list_downloaded_models', {
    storagePath,
  });

  return rows.map((row) => ({
    id: row.id,
    name: row.name,
    pipeline_tag: row.pipeline_tag,
    sizeBytes: row.size_bytes,
    lastUsed: row.last_used,
    storagePath: row.storage_path,
  }));
}

export async function deleteDownloadedModel(modelId: string, storagePath: string): Promise<void> {
  await invoke('unload_model');
  await invoke('delete_downloaded_model', {
    modelId,
    storagePath,
  });
}
