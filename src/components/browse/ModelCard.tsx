import { memo } from 'react';
import { Download } from 'lucide-react';
import type { HFModel } from '../../stores/appStore';
import { useAppStore } from '../../stores/appStore';
import Badge from '../shared/Badge';
import {
  estimateModelSize,
  formatBytes,
  formatDownloads,
  isModelSizeEstimated,
} from '../../services/huggingfaceApi';
import { getCompatibility } from '../../utils/ramEstimation';

interface ModelCardProps {
  model: HFModel;
}

const COMPAT_CONFIG = {
  compatible: { dot: 'var(--success)', label: 'Compatible' },
  tight: { dot: 'var(--warning)', label: 'May be tight' },
  'too-large': { dot: 'var(--error)', label: 'Too large' },
  unknown: { dot: 'var(--text-muted)', label: '' },
};

function ModelCard({ model }: ModelCardProps) {
  const navigateToModel = useAppStore((s) => s.navigateToModel);
  const totalRam = useAppStore((s) => s.systemInfo.totalRam);

  const sizeBytes = estimateModelSize(model);
  const sizeEstimated = isModelSizeEstimated(model);
  const compat = getCompatibility(model, totalRam);
  const compatInfo = COMPAT_CONFIG[compat];

  const [org, ...nameParts] = (model.modelId ?? model.id).split('/');
  const modelName = nameParts.join('/') || org;
  const orgName = nameParts.length ? org : null;

  const description =
    model.description ??
    (model.cardData as { text?: string } | undefined)?.text ??
    '';

  return (
    <article
      onClick={() => navigateToModel(model.modelId ?? model.id)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          navigateToModel(model.modelId ?? model.id);
        }
      }}
      aria-label={`${model.modelId ?? model.id} — ${model.pipeline_tag ?? 'model'}`}
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '16px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        transition: 'border-color 100ms',
        outline: 'none',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor =
          'rgba(255,107,53,0.5)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)';
      }}
      onFocus={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent-primary)';
      }}
      onBlur={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)';
      }}
    >
      {/* Pipeline badge */}
      <Badge pipeline_tag={model.pipeline_tag} />

      {/* Model name */}
      <div>
        {orgName && (
          <div
            style={{
              fontFamily: '"Inter", sans-serif',
              fontSize: '12px',
              color: 'var(--text-muted)',
              marginBottom: '2px',
            }}
          >
            {orgName}
          </div>
        )}
        <div
          style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: '15px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={modelName}
        >
          {modelName}
        </div>
      </div>

      {/* Description */}
      {description && (
        <p
          style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: '13px',
            color: 'var(--text-secondary)',
            lineHeight: '1.4',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            margin: 0,
            flex: 1,
          }}
        >
          {description}
        </p>
      )}

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginTop: 'auto',
        }}
      >
        {model.downloads > 0 && (
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: '12px',
              color: 'var(--text-muted)',
            }}
          >
            <Download size={12} strokeWidth={1.5} />
            {formatDownloads(model.downloads)}
          </span>
        )}
        {sizeBytes > 0 && (
          <>
            <span style={{ color: 'var(--border)', fontSize: '12px' }}>·</span>
            <span
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '12px',
                color: 'var(--text-muted)',
              }}
            >
              {sizeEstimated ? '~' : ''}
              {formatBytes(sizeBytes)}
            </span>
          </>
        )}

        {/* Compatibility indicator */}
        {compatInfo.label && (
          <span
            style={{
              marginLeft: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: '11px',
              color: 'var(--text-muted)',
              flexShrink: 0,
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: compatInfo.dot,
                display: 'inline-block',
                flexShrink: 0,
              }}
            />
            {compatInfo.label}
          </span>
        )}
      </div>
    </article>
  );
}

export default memo(ModelCard);
