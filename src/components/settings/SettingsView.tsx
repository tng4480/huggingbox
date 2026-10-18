import { useCallback, useEffect, useState } from 'react';
import { Eye, EyeOff, ExternalLink, RefreshCw, Trash2 } from 'lucide-react';
import { open } from '@tauri-apps/plugin-dialog';
import { useAppStore } from '../../stores/appStore';
import { confirmDialog } from '../../services/dialogs';
import { listGpus, type GpuInfo } from '../../services/gpuInfo';
import {
  deleteModelEnvironment,
  getModelEnvironmentSize,
  listModelEnvironments,
  type ModelEnvironment,
} from '../../services/modelEnvironments';

interface FieldProps {
  label: string;
  helper?: string;
  helperLink?: { text: string; href: string };
  children: React.ReactNode;
}

function Field({ label, helper, helperLink, children }: FieldProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
      <label
        style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '13px',
          fontWeight: 500,
          color: 'var(--text-primary)',
        }}
      >
        {label}
      </label>
      {children}
      {(helper || helperLink) && (
        <p
          style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: '13px',
            color: 'var(--text-muted)',
            margin: 0,
          }}
        >
          {helper}{' '}
          {helperLink && (
            <a
              href={helperLink.href}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: 'var(--accent-secondary)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '2px',
              }}
            >
              {helperLink.text}
              <ExternalLink size={11} strokeWidth={1.5} />
            </a>
          )}
        </p>
      )}
    </div>
  );
}

function PasswordInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  const [show, setShow] = useState(false);

  return (
    <div style={{ position: 'relative' }}>
      <input
        type={show ? 'text' : 'password'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%',
          height: '36px',
          backgroundColor: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '0 36px 0 12px',
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '13px',
          color: 'var(--text-primary)',
          outline: 'none',
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = 'var(--accent-primary)';
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = 'var(--border)';
        }}
      />
      <button
        type="button"
        onClick={() => setShow((s) => !s)}
        aria-label={show ? 'Hide' : 'Show'}
        style={{
          position: 'absolute',
          right: '8px',
          top: '50%',
          transform: 'translateY(-50%)',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          padding: '4px',
        }}
      >
        {show ? <EyeOff size={14} strokeWidth={1.5} /> : <Eye size={14} strokeWidth={1.5} />}
      </button>
    </div>
  );
}

const SECTION_STYLE: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 'var(--space-lg)',
  paddingBottom: 'var(--space-2xl)',
  borderBottom: '1px solid var(--border)',
};

const SECTION_TITLE: React.CSSProperties = {
  fontFamily: '"Inter", sans-serif',
  fontSize: '16px',
  fontWeight: 600,
  color: 'var(--text-primary)',
  margin: '0 0 var(--space-xs) 0',
};

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** idx;
  return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

export default function SettingsView() {
  const settings = useAppStore((s) => s.settings);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const [storageDraft, setStorageDraft] = useState(settings.modelStoragePath);
  const [envStorageDraft, setEnvStorageDraft] = useState(settings.envStoragePath);
  const [pickerError, setPickerError] = useState<string | null>(null);
  const [gpuOptions, setGpuOptions] = useState<GpuInfo[]>([]);
  const [gpuLoadError, setGpuLoadError] = useState<string | null>(null);
  const [modelEnvs, setModelEnvs] = useState<ModelEnvironment[]>([]);
  const [envLoading, setEnvLoading] = useState(false);
  const [envError, setEnvError] = useState<string | null>(null);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);

  useEffect(() => {
    setStorageDraft(settings.modelStoragePath);
  }, [settings.modelStoragePath]);

  useEffect(() => {
    setEnvStorageDraft(settings.envStoragePath);
  }, [settings.envStoragePath]);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        setGpuLoadError(null);
        const gpus = await listGpus();
        if (active) {
          setGpuOptions(gpus);
        }
      } catch (error) {
        if (active) {
          setGpuLoadError(String(error));
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const refreshModelEnvs = useCallback(async () => {
    try {
      setEnvLoading(true);
      setEnvError(null);
      const rows = await listModelEnvironments(settings.envStoragePath, { includeSizes: false });
      setModelEnvs(rows);
    } catch (error) {
      setEnvError(String(error));
    } finally {
      setEnvLoading(false);
    }
  }, [settings.envStoragePath]);

  useEffect(() => {
    void refreshModelEnvs();
  }, [settings.envStoragePath]);

  useEffect(() => {
    const pending = modelEnvs
      .filter((env) => env.sizeBytes === null)
      .map((env) => env.modelId);
    if (pending.length === 0) return undefined;

    let cancelled = false;
    void (async () => {
      for (const modelId of pending) {
        if (cancelled) break;
        try {
          const sizeBytes = await getModelEnvironmentSize(modelId, settings.envStoragePath);
          if (cancelled) break;
          setModelEnvs((current) =>
            current.map((env) => (env.modelId === modelId ? { ...env, sizeBytes } : env))
          );
        } catch {
          // Keep Settings page responsive; size is optional metadata.
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [modelEnvs, settings.envStoragePath]);

  function normalizeStoragePath(input: string): string {
    const trimmed = input.trim();
    if (!trimmed) return settings.modelStoragePath;
    return trimmed;
  }

  function saveStoragePath(value: string) {
    const normalized = normalizeStoragePath(value);
    updateSettings({ modelStoragePath: normalized });
    setStorageDraft(normalized);
  }

  function saveEnvStoragePath(value: string) {
    updateSettings({ envStoragePath: value.trim() });
    setEnvStorageDraft(value.trim());
  }

  async function handleChooseDirectory() {
    try {
      setPickerError(null);
      const selected = await open({
        directory: true,
        multiple: false,
        defaultPath: settings.modelStoragePath,
      });
      if (typeof selected === 'string' && selected.trim()) {
        saveStoragePath(selected);
      }
    } catch {
      setPickerError('Folder picker is unavailable. Enter a full path manually below.');
    }
  }

  async function handleDeleteEnv(modelId: string) {
    const confirmed = await confirmDialog(
      `Delete the isolated Python environment for:\n${modelId}\n\nIt will be recreated on next run.`
    );
    if (!confirmed) return;

    try {
      setDeletingModelId(modelId);
      await deleteModelEnvironment(modelId, settings.envStoragePath);
      await refreshModelEnvs();
    } catch (error) {
      setEnvError(String(error));
    } finally {
      setDeletingModelId(null);
    }
  }

  return (
    <div
      style={{
        height: '100%',
        overflowY: 'auto',
        padding: 'var(--space-2xl)',
      }}
    >
      <div
        style={{
          maxWidth: '640px',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-2xl)',
        }}
      >
        <h1
          style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: '20px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            margin: 0,
          }}
        >
          Settings
        </h1>



        {/* Hugging Face Token */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Hugging Face</h2>
          <Field
            label="Hugging Face Token (optional)"
            helper="Required for gated models (e.g., Llama, Mistral). Get your token at"
            helperLink={{ text: 'huggingface.co/settings/tokens', href: 'https://huggingface.co/settings/tokens' }}
          >
            <PasswordInput
              value={settings.hfToken}
              onChange={(v) => updateSettings({ hfToken: v })}
              placeholder="hf_..."
            />
          </Field>
        </div>

        {/* Code Generation */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Code Generation</h2>
          <Field
            label="Default Generator"
            helper="Auto Runner works locally. Codex CLI and Claude Code use your existing CLI sign-in and account usage limits; no API key is required in HuggingBox. Model documentation, and code/error logs for repair, are sent to the selected provider."
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              {([
                { value: 'autorunner', label: 'Auto Runner (default)' },
                { value: 'codex-cli', label: 'Codex CLI (existing sign-in)' },
                { value: 'claude-code', label: 'Claude Code (existing sign-in)' },
                { value: 'claude-sonnet', label: 'Claude Sonnet (API key)' },
              ] as const).map((option) => (
                <label
                  key={option.value}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-sm)',
                    cursor: 'pointer',
                    fontFamily: '"Inter", sans-serif',
                    fontSize: '14px',
                    color: 'var(--text-primary)',
                  }}
                >
                  <input
                    type="radio"
                    name="code-generator"
                    value={option.value}
                    checked={settings.codeGenerationProvider === option.value}
                    onChange={() => updateSettings({ codeGenerationProvider: option.value })}
                    style={{ accentColor: 'var(--accent-primary)' }}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </Field>
          <Field label="Codex CLI executable (optional)" helper="Install Codex CLI and run codex login in a terminal. Leave blank for automatic detection.">
            <input className="cli-path-input" aria-label="Codex CLI executable" value={settings.codexCliPath || ''} onChange={e => updateSettings({codexCliPath: e.target.value})} placeholder="Absolute path to codex" />
          </Field>
          <Field label="Claude Code executable (optional)" helper="Install Claude Code and run claude in a terminal to sign in. Leave blank for automatic detection.">
            <input className="cli-path-input" aria-label="Claude Code executable" value={settings.claudeCodePath || ''} onChange={e => updateSettings({claudeCodePath: e.target.value})} placeholder="Absolute path to claude" />
          </Field>
          <Field
            label="Anthropic API Key"
            helper="Required before Claude Sonnet can be used for code generation."
            helperLink={{ text: 'console.anthropic.com', href: 'https://console.anthropic.com/' }}
          >
            <PasswordInput
              value={settings.claudeApiKey}
              onChange={(v) => updateSettings({ claudeApiKey: v })}
              placeholder="sk-ant-..."
            />
          </Field>
          <Field
            label="Claude Dependency Install"
            helper="If enabled, HuggingBox will offer to install Claude-suggested Python packages into the model environment after generation."
          >
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                cursor: 'pointer',
                fontFamily: '"Inter", sans-serif',
                fontSize: '14px',
                color: 'var(--text-primary)',
              }}
            >
              <input
                type="checkbox"
                checked={settings.claudeAutoInstallDependencies}
                onChange={(e) => updateSettings({ claudeAutoInstallDependencies: e.target.checked })}
                style={{ accentColor: 'var(--accent-primary)' }}
              />
              Offer dependency installation after Claude generation
            </label>
          </Field>
        </div>

        {/* Model Storage */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Model Storage</h2>
          <Field label="Model Storage Directory">
            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
              <input
                type="text"
                value={storageDraft}
                onChange={(e) => setStorageDraft(e.target.value)}
                onBlur={(e) => saveStoragePath(e.target.value)}
                style={{
                  flex: 1,
                  height: '36px',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  padding: '0 12px',
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '13px',
                  color: 'var(--text-primary)',
                }}
              />
              <button
                onClick={handleChooseDirectory}
                style={{
                  height: '36px',
                  padding: '0 16px',
                  backgroundColor: 'transparent',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '13px',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
              >
                Change
              </button>
            </div>
            {pickerError && (
              <p
                style={{
                  margin: '6px 0 0 0',
                  fontFamily: '"Inter", sans-serif',
                  fontSize: '12px',
                  color: 'var(--warning)',
                }}
              >
                {pickerError}
              </p>
            )}
          </Field>
        </div>

        {/* Environment Storage */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Environment Storage</h2>
          <Field
            label="Python Environments Directory (optional)"
            helper="Where isolated Python venvs are created per model. Leave blank to use the default AppData location."
          >
            <input
              type="text"
              value={envStorageDraft}
              onChange={(e) => setEnvStorageDraft(e.target.value)}
              onBlur={(e) => saveEnvStoragePath(e.target.value)}
              placeholder="e.g. D:\HuggingBox\envs  (blank = auto)"
              style={{
                width: '100%',
                height: '36px',
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '0 12px',
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '13px',
                color: 'var(--text-primary)',
                boxSizing: 'border-box',
              }}
            />
          </Field>
        </div>


        {/* Device */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Inference Device</h2>
          <Field label="Preferred Device">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              {(['auto', 'cpu', 'cuda'] as const).map((d) => (
                <label
                  key={d}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-sm)',
                    cursor: 'pointer',
                    fontFamily: '"Inter", sans-serif',
                    fontSize: '14px',
                    color: 'var(--text-primary)',
                  }}
                >
                  <input
                    type="radio"
                    name="device"
                    value={d}
                    checked={settings.preferredDevice === d}
                    onChange={() => updateSettings({ preferredDevice: d })}
                    style={{ accentColor: 'var(--accent-primary)' }}
                  />
                  {d === 'auto' ? 'Auto (recommended)' : d === 'cpu' ? 'CPU only' : 'GPU (CUDA)'}
                </label>
              ))}
            </div>
          </Field>
          <Field
            label="GPU Selection"
            helper={settings.preferredDevice === 'cpu'
              ? 'GPU selection is disabled while CPU-only mode is active.'
              : 'Pick a specific GPU when multiple CUDA GPUs are available.'}
          >
            <select
              value={settings.selectedGpuId ?? ''}
              disabled={settings.preferredDevice === 'cpu' || gpuOptions.length === 0}
              onChange={(e) => {
                const value = e.target.value;
                updateSettings({ selectedGpuId: value || null });
              }}
              style={{
                width: '100%',
                height: '36px',
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '0 12px',
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '13px',
                color: 'var(--text-primary)',
              }}
            >
              <option value="">Auto-select GPU</option>
              {gpuOptions.map((gpu) => (
                <option key={gpu.id} value={gpu.id}>
                  {gpu.name}{gpu.vramGb ? ` (${gpu.vramGb} GB)` : ''}
                </option>
              ))}
            </select>
            {gpuLoadError && (
              <p
                style={{
                  margin: '6px 0 0 0',
                  fontFamily: '"Inter", sans-serif',
                  fontSize: '12px',
                  color: 'var(--warning)',
                }}
              >
                Could not query GPUs: {gpuLoadError}
              </p>
            )}
          </Field>
        </div>

        {/* Model Environments */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Model Environments</h2>
          <Field
            label="Per-Model Python venvs"
            helper="Each model has an isolated environment. Delete to reclaim space or rebuild a broken env."
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
              <span
                style={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '12px',
                  color: 'var(--text-muted)',
                }}
              >
                {modelEnvs.length} environments
              </span>
              <button
                onClick={() => void refreshModelEnvs()}
                disabled={envLoading}
                style={{
                  height: '30px',
                  padding: '0 10px',
                  backgroundColor: 'transparent',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '12px',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <RefreshCw size={12} strokeWidth={1.5} />
                Refresh
              </button>
            </div>

            <div
              style={{
                marginTop: '8px',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                overflow: 'hidden',
                maxHeight: '220px',
                overflowY: 'auto',
              }}
            >
              {envLoading && (
                <p style={{ margin: 0, padding: '10px 12px', fontFamily: '"Inter", sans-serif', fontSize: '13px', color: 'var(--text-muted)' }}>
                  Loading environments...
                </p>
              )}
              {!envLoading && modelEnvs.length === 0 && (
                <p style={{ margin: 0, padding: '10px 12px', fontFamily: '"Inter", sans-serif', fontSize: '13px', color: 'var(--text-muted)' }}>
                  No model environments created yet.
                </p>
              )}
              {!envLoading && modelEnvs.map((env) => (
                <div
                  key={env.modelId}
                  style={{
                    padding: '10px 12px',
                    borderTop: '1px solid var(--border)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <p
                      style={{
                        margin: 0,
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: '12px',
                        color: 'var(--text-primary)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {env.modelId}
                    </p>
                    <p
                      style={{
                        margin: '2px 0 0 0',
                        fontFamily: '"Inter", sans-serif',
                        fontSize: '12px',
                        color: 'var(--text-muted)',
                      }}
                    >
                      {env.sizeBytes === null ? 'Calculating size...' : formatBytes(env.sizeBytes)}
                    </p>
                  </div>
                  <button
                    onClick={() => void handleDeleteEnv(env.modelId)}
                    disabled={deletingModelId === env.modelId}
                    style={{
                      height: '28px',
                      padding: '0 10px',
                      backgroundColor: 'transparent',
                      border: '1px solid var(--border)',
                      borderRadius: '4px',
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: '12px',
                      color: 'var(--error)',
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      flexShrink: 0,
                    }}
                  >
                    <Trash2 size={12} strokeWidth={1.5} />
                    Delete
                  </button>
                </div>
              ))}
            </div>
            {envError && (
              <p
                style={{
                  margin: '6px 0 0 0',
                  fontFamily: '"Inter", sans-serif',
                  fontSize: '12px',
                  color: 'var(--warning)',
                }}
              >
                Could not manage environments: {envError}
              </p>
            )}
          </Field>
        </div>

        {/* Appearance */}
        <div style={SECTION_STYLE}>
          <h2 style={SECTION_TITLE}>Appearance</h2>
          <Field label="Theme">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              {(['dark', 'light'] as const).map((t) => (
                <label
                  key={t}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-sm)',
                    cursor: 'pointer',
                    fontFamily: '"Inter", sans-serif',
                    fontSize: '14px',
                    color: 'var(--text-primary)',
                  }}
                >
                  <input
                    type="radio"
                    name="theme"
                    value={t}
                    checked={settings.theme === t}
                    onChange={() => {
                      updateSettings({ theme: t });
                      document.documentElement.setAttribute('data-theme', t);
                    }}
                    style={{ accentColor: 'var(--accent-primary)' }}
                  />
                  {t === 'dark' ? 'Dark (default)' : 'Light'}
                </label>
              ))}
            </div>
          </Field>
        </div>

        {/* About */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <h2 style={SECTION_TITLE}>About</h2>
          <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '13px', color: 'var(--text-muted)' }}>
            HuggingBox v0.3.0
          </p>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              fontFamily: '"Inter", sans-serif',
              fontSize: '14px',
              color: 'var(--accent-secondary)',
            }}
          >
            View on GitHub
            <ExternalLink size={12} strokeWidth={1.5} />
          </a>
        </div>
      </div>
    </div>
  );
}
