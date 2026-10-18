import { useEffect, useState } from 'react';
import { PackageOpen } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';
import Button from '../shared/Button';
import { confirmDialog } from '../../services/dialogs';
import { deleteDownloadedModel, listDownloadedModels } from '../../services/modelStorage';

export default function MyModelsView() {
  const downloadedModels = useAppStore((s) => s.downloadedModels);
  const setCurrentView = useAppStore((s) => s.setCurrentView);
  const setDownloadedModels = useAppStore((s) => s.setDownloadedModels);
  const storagePath = useAppStore((s) => s.settings.modelStoragePath);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);
  const navigateToModel = useAppStore((s) => s.navigateToModel);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const rows = await listDownloadedModels(storagePath);
        if (mounted) {
          setDownloadedModels(rows);
          setLoadError(null);
        }
      } catch {
        if (mounted) setLoadError('Could not load downloaded models from storage.');
      }
    })();
    return () => {
      mounted = false;
    };
  }, [setDownloadedModels, storagePath]);

  async function refreshDownloadedModels() {
    try {
      const rows = await listDownloadedModels(storagePath);
      setDownloadedModels(rows);
      setLoadError(null);
    } catch {
      setLoadError('Could not load downloaded models from storage.');
    }
  }

  async function handleDeleteModel(modelId: string) {
    const confirmed = await confirmDialog(
      `Remove cached weights for:\n${modelId}\n\nThis removes only managed model weights. Your scripts, results and legacy files are preserved.`
    );
    if (!confirmed) return;

    try {
      setDeletingModelId(modelId);
      await deleteDownloadedModel(modelId, storagePath);
      await refreshDownloadedModels();
    } catch (error) {
      setLoadError(`Could not remove cached weights: ${String(error)}`);
    } finally {
      setDeletingModelId(null);
    }
  }

  if (downloadedModels.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          gap: 'var(--space-xl)',
          color: 'var(--text-muted)',
          fontFamily: '"Inter", sans-serif',
        }}
      >
        <PackageOpen size={48} strokeWidth={1} color="var(--text-muted)" />
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 'var(--space-xs)' }}>
            No models downloaded yet
          </p>
          <p style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
            {loadError ?? 'Browse models to get started.'}
          </p>
        </div>
        <Button
          variant="primary"
          onClick={() => setCurrentView('browse')}
          icon={
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          }
        >
          Browse Models
        </Button>
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: 'var(--space-md) var(--space-xl)',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <h2
          style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: '16px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            margin: 0,
          }}
        >
          My Models
        </h2>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-xl)' }}>
        {downloadedModels.map((model) => (
          <div
            key={model.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: 'var(--space-md) var(--space-lg)',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              marginBottom: 'var(--space-sm)',
            }}
          >
            <div>
              <div style={{ fontFamily: '"Inter", sans-serif', fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                {model.name}
              </div>
              <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                {model.pipeline_tag} · {(model.sizeBytes / 1024 ** 3).toFixed(1)} GB · Last used: {model.lastUsed}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
              <Button
                variant="secondary"
                style={{ height: '30px', fontSize: '12px' }}
                onClick={() => navigateToModel(model.id)}
              >
                Run
              </Button>
              <Button
                variant="danger"
                style={{ height: '30px', fontSize: '12px' }}
                onClick={() => void handleDeleteModel(model.id)}
                disabled={deletingModelId === model.id}
              >
                Delete
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
