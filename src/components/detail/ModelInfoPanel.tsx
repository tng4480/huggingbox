import { useState } from 'react';
import { ExternalLink, Download, HardDrive, Cpu } from 'lucide-react';
import type { HFModelDetail } from '../../stores/appStore';
import { useAppStore } from '../../stores/appStore';
import Badge from '../shared/Badge';
import Button from '../shared/Button';
import {
  estimateModelSize,
  formatBytes,
  formatDownloads,
  getModelFormatInfo,
  getGgufFiles,
  resolveGgufFile,
} from '../../services/huggingfaceApi';
import { estimateRamBytes, getCompatibility } from '../../utils/ramEstimation';

interface ModelInfoPanelProps {
  model: HFModelDetail;
  onUseModel: () => void;
  codeGenerationError?: string | null;
}

const COMPAT_CONFIG = {
  compatible: { color: 'var(--success)', dot: 'var(--success)', label: 'Compatible with your device' },
  tight: { color: 'var(--warning)', dot: 'var(--warning)', label: 'May be tight — close other apps' },
  'too-large': { color: 'var(--error)', dot: 'var(--error)', label: 'Too large for your device' },
  unknown: { color: 'var(--text-muted)', dot: 'var(--text-muted)', label: 'Compatibility unknown' },
};

const MAX_DESC = 300;

export default function ModelInfoPanel({
  model,
  onUseModel,
  codeGenerationError,
}: ModelInfoPanelProps) {
  const [descExpanded, setDescExpanded] = useState(false);
  const totalRam = useAppStore((s) => s.systemInfo.totalRam);

  const selectedGguf = useAppStore((s) => s.selectedGguf);
  const setSelectedGguf = useAppStore((s) => s.setSelectedGguf);
  const ggufFiles = getGgufFiles(model);
  const activeGguf = resolveGgufFile(model, selectedGguf);

  const sizeBytes = estimateModelSize(model, selectedGguf);
  const ramEstBytes = estimateRamBytes(model, selectedGguf);
  const compat = getCompatibility(model, totalRam, selectedGguf);
  const compatInfo = COMPAT_CONFIG[compat];

  const [org, ...nameParts] = (model.modelId ?? model.id).split('/');
  const modelName = nameParts.join('/') || org;
  const orgName = nameParts.length ? org : null;

  const rawDesc = model.description ?? '';
  const truncated = rawDesc.length > MAX_DESC && !descExpanded;
  const displayDesc = truncated ? rawDesc.slice(0, MAX_DESC) + '...' : rawDesc;

  const formatInfo = getModelFormatInfo(model);
  const formats = [
    formatInfo.hasGguf && 'GGUF',
    formatInfo.hasOnnx && 'ONNX',
    formatInfo.hasSafetensors && 'SafeTensors',
    formatInfo.hasPytorch && 'PyTorch',
  ]
    .filter(Boolean)
    .join(', ') || 'Unknown';

  const hfUrl = `https://huggingface.co/${model.modelId ?? model.id}`;

  const metaStyle = {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '13px',
    color: 'var(--text-primary)',
    fontWeight: 500,
  } as const;

  const metaLabelStyle = {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '11px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
    marginBottom: '2px',
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-xl)',
        padding: 'var(--space-2xl)',
        maxWidth: '800px',
        width: '100%',
        margin: '0 auto',
      }}
    >
      {/* Header: badge + name */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        <Badge pipeline_tag={model.pipeline_tag} />
        {orgName && (
          <div style={{ fontFamily: '"Inter", sans-serif', fontSize: '14px', color: 'var(--text-muted)' }}>
            {orgName}
          </div>
        )}
        <h1
          style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: '24px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            margin: 0,
          }}
        >
          {modelName}
        </h1>
      </div>

      {/* Description */}
      {displayDesc && (
        <div style={{ fontFamily: '"Inter", sans-serif', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
          {displayDesc}
          {rawDesc.length > MAX_DESC && (
            <button
              onClick={() => setDescExpanded((x) => !x)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--accent-secondary)',
                cursor: 'pointer',
                fontSize: '14px',
                fontFamily: '"Inter", sans-serif',
                padding: '0 4px',
              }}
            >
              {descExpanded ? 'less' : 'more'}
            </button>
          )}
        </div>
      )}

      {/* Metadata grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 'var(--space-sm)',
        }}
      >
        {[
          {
            label: 'Size',
            icon: <HardDrive size={14} strokeWidth={1.5} />,
            value: sizeBytes > 0 ? formatBytes(sizeBytes) : '—',
          },
          {
            label: 'Downloads',
            icon: <Download size={14} strokeWidth={1.5} />,
            value: model.downloads > 0 ? formatDownloads(model.downloads) : '—',
          },
          {
            label: 'Format',
            icon: null,
            value: formats,
          },
          {
            label: 'RAM Est.',
            icon: <Cpu size={14} strokeWidth={1.5} />,
            value: ramEstBytes > 0 ? `~${formatBytes(ramEstBytes)}` : '—',
          },
        ].map(({ label, icon, value }) => (
          <div
            key={label}
            style={{
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: 'var(--space-md)',
            }}
          >
            <div style={{ ...metaLabelStyle, display: 'flex', alignItems: 'center', gap: '4px' }}>
              {icon}
              {label}
            </div>
            <div style={metaStyle}>{value}</div>
          </div>
        ))}
      </div>

      {/* Quantization picker (GGUF repos ship many alternative quantizations of one model) */}
      {ggufFiles.length > 1 && activeGguf && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <label htmlFor="gguf-select" style={metaLabelStyle}>
            Quantization ({ggufFiles.length} files)
          </label>
          <select
            id="gguf-select"
            value={activeGguf.name}
            onChange={(e) => setSelectedGguf(e.target.value)}
            style={{
              ...metaStyle,
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '8px 10px',
              width: '100%',
            }}
          >
            {ggufFiles.map((f) => (
              <option key={f.name} value={f.name}>
                {f.quant}
                {f.size > 0 ? ` — ${formatBytes(f.size)}` : ''}
                {` — ${f.name}`}
              </option>
            ))}
          </select>
          <div style={{ fontFamily: '"Inter", sans-serif', fontSize: '12px', color: 'var(--text-muted)' }}>
            Size and RAM estimate are for the selected file only. This file is the one that will be downloaded and run.
          </div>
        </div>
      )}

      {/* Compatibility indicator */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-sm)',
            fontFamily: '"Inter", sans-serif',
            fontSize: '14px',
            color: compatInfo.color,
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: compatInfo.dot,
              display: 'inline-block',
              flexShrink: 0,
            }}
          />
          {compatInfo.label}
        </div>
        {formatInfo.recommendationReason && (
          <div
            style={{
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '10px 12px',
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
                marginBottom: '4px',
              }}
            >
              Recommended Runtime
            </div>
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '13px',
                color: 'var(--text-primary)',
                marginBottom: '4px',
              }}
            >
              {formatInfo.recommendedRuntime}
            </div>
            <div
              style={{
                fontFamily: '"Inter", sans-serif',
                fontSize: '13px',
                color: 'var(--text-secondary)',
                lineHeight: 1.5,
              }}
            >
              {formatInfo.recommendationReason}
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-md)' }}>
        <Button
          variant="primary"
          onClick={onUseModel}
          style={{ minWidth: '200px', justifyContent: 'center', height: '44px', fontSize: '14px' }}
        >
          Use This Model
        </Button>

        {codeGenerationError && (
          <p
            style={{
              margin: 0,
              maxWidth: '560px',
              textAlign: 'center',
              fontFamily: '"Inter", sans-serif',
              fontSize: '13px',
              color: 'var(--error)',
            }}
          >
            {codeGenerationError}
          </p>
        )}

        <Button
          variant="ghost"
          href={hfUrl}
          target="_blank"
          rel="noopener noreferrer"
          icon={<ExternalLink size={14} strokeWidth={1.5} />}
          aria-label={`View ${model.modelId ?? model.id} on Hugging Face`}
        >
          View on Hugging Face
        </Button>
      </div>
    </div>
  );
}
