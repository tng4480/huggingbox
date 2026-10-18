import { useAppStore } from '../../stores/appStore';

function formatRam(bytes: number): string {
  return (bytes / (1024 ** 3)).toFixed(1);
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

export default function StatusBar() {
  const systemInfo = useAppStore((s) => s.systemInfo);
  const labJobLabel = useAppStore(s => s.labJobLabel);
  const executionState = useAppStore((s) => s.executionState);
  const executionElapsed = useAppStore((s) => s.executionElapsed);
  const executionStats = useAppStore((s) => s.executionStats);
  const activeExecutionModelId = useAppStore((s) => s.activeExecutionModelId);
  const navigateToModel = useAppStore((s) => s.navigateToModel);

  function getPythonStatus(): { label: string; color: string } {
    if (executionState === 'installing') {
      return { label: 'Installing packages...', color: 'var(--warning)' };
    }
    if (!systemInfo.pythonReady) {
      return { label: 'Environment Error', color: 'var(--error)' };
    }
    return { label: 'Python Ready', color: 'var(--success)' };
  }

  const pythonStatus = getPythonStatus();
  const usedRam = systemInfo.totalRam - systemInfo.availableRam;
  const processRam = executionStats.processRssBytes;
  const isExecutionClickable = Boolean(activeExecutionModelId);
  const isActiveExecution =
    executionState === 'running' || executionState === 'installing' || executionState === 'downloading';

  function getExecutionLabel(): string {
    switch (executionState) {
      case 'running':
        return `Running... ${formatElapsed(executionElapsed)}`;
      case 'installing':
        return `Running... ${formatElapsed(executionElapsed)} (installing packages)`;
      case 'downloading':
        return `Running... ${formatElapsed(executionElapsed)} (downloading model)`;
      case 'completed':
        return `Completed (${formatElapsed(executionElapsed)})`;
      case 'error':
        return 'Error';
      case 'cancelled':
        return 'Cancelled';
      case 'stopping':
        return 'Stopping…';
      default:
        return 'Idle';
    }
  }

  function getExecutionColor(): string {
    switch (executionState) {
      case 'running':
      case 'installing':
      case 'downloading':
        return 'var(--accent-primary)';
      case 'completed':
        return 'var(--success)';
      case 'error':
        return 'var(--error)';
      default:
        return 'var(--text-secondary)';
    }
  }

  const mono: React.CSSProperties = {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '12px',
    color: 'var(--text-secondary)',
  };

  const divider = (
    <span style={{ color: 'var(--border)', margin: '0 var(--space-sm)' }}>|</span>
  );

  return (
    <footer
      style={{
        height: '28px',
        minHeight: '28px',
        backgroundColor: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 var(--space-lg)',
        gap: 0,
        flexShrink: 0,
      }}
      aria-live="polite"
      aria-label="Status bar"
    >
      <span style={mono}>
        {isActiveExecution && processRam
          ? `Proc RAM: ${formatRam(processRam)} GB`
          : 'RAM: '}
        {!isActiveExecution && systemInfo.totalRam > 0 ? (
          <>
            {formatRam(usedRam)}/{formatRam(systemInfo.totalRam)} GB
          </>
        ) : isActiveExecution && processRam ? null : (
          '...'
        )}
      </span>

      {divider}

      <span style={mono}>
        {isActiveExecution && executionStats.executionRuntime
          ? `Runtime: ${executionStats.executionRuntime}${executionStats.executionProvider ? ` (${executionStats.executionProvider.replace('ExecutionProvider', '')})` : ''}`
          : systemInfo.gpuName
            ? `GPU: ${systemInfo.gpuName}${systemInfo.gpuVram ? ` ${systemInfo.gpuVram}GB` : ''}`
            : 'No GPU detected'}
      </span>

      {isActiveExecution && executionStats.tokensPerSecond ? (
        <>
          {divider}
          <span style={mono}>{executionStats.tokensPerSecond.toFixed(1)} tok/s</span>
        </>
      ) : isActiveExecution && executionStats.inferenceSeconds ? (
        <>
          {divider}
          <span style={mono}>{executionStats.inferenceSeconds.toFixed(2)}s infer</span>
        </>
      ) : null}

      {divider}

      <button
        onClick={() => {
          if (labJobLabel) { useAppStore.getState().setCurrentView('lab'); return; }
          if (!activeExecutionModelId) return;
          navigateToModel(activeExecutionModelId, { preserveWorkspace: true });
        }}
        disabled={!isExecutionClickable && !labJobLabel}
        title={
          labJobLabel ? 'Open Model Lab' : isExecutionClickable
            ? `Open execution workspace for ${activeExecutionModelId}`
            : 'No active execution workspace'
        }
        style={{
          ...mono,
          flex: 1,
          textAlign: 'center',
          color: getExecutionColor(),
          background: 'none',
          border: 'none',
          cursor: isExecutionClickable ? 'pointer' : 'default',
          opacity: isExecutionClickable ? 1 : 0.85,
          padding: 0,
        }}
      >
        {labJobLabel || getExecutionLabel()}
      </button>

      <span style={{ ...mono, color: pythonStatus.color }}>
        {pythonStatus.label}
      </span>
    </footer>
  );
}
