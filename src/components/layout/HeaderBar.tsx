import { Sun, Moon, Box } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

type NavView = 'browse' | 'my-models' | 'settings';

const NAV_TABS: { label: string; view: NavView }[] = [
  { label: 'Browse', view: 'browse' },
  { label: 'My Models', view: 'my-models' },
  { label: 'Settings', view: 'settings' },
];

export default function HeaderBar() {
  const currentView = useAppStore((s) => s.currentView);
  const setCurrentView = useAppStore((s) => s.setCurrentView);
  const selectedModelId = useAppStore(s => s.selectedModelId);
  const theme = useAppStore((s) => s.settings.theme);
  const setTheme = useAppStore((s) => s.setTheme);

  const isActive = (view: NavView) =>
    currentView === view || (currentView === 'model-detail' && view === 'browse');

  function handleNavClick(view: NavView) {
    setCurrentView(view);
  }

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.setAttribute('data-theme', next);
  }

  return (
    <header
      style={{
        height: '48px',
        minHeight: '48px',
        backgroundColor: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 var(--space-lg)',
        gap: 'var(--space-lg)',
        userSelect: 'none',
        flexShrink: 0,
      }}
    >
      {/* Logo */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-sm)',
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: '16px',
          fontWeight: 500,
          color: 'var(--accent-primary)',
          letterSpacing: '-0.02em',
          flexShrink: 0,
        }}
      >
        <Box size={18} strokeWidth={1.5} color="var(--accent-primary)" />
        HuggingBox
      </div>

      {/* Nav tabs */}
      <nav
        style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', flex: 1 }}
        aria-label="Main navigation"
      >
        {selectedModelId && currentView !== 'model-detail' && <button className="return-workspace" onClick={() => useAppStore.getState().navigateToModel(selectedModelId)}>Return to workspace</button>}
        {NAV_TABS.map(({ label, view }) => (
          <button
            key={view}
            onClick={() => handleNavClick(view)}
            aria-current={isActive(view) ? 'page' : undefined}
            style={{
              height: '48px',
              padding: '0 var(--space-md)',
              background: 'none',
              border: 'none',
              borderBottom: isActive(view)
                ? '2px solid var(--accent-primary)'
                : '2px solid transparent',
              borderTop: '2px solid transparent',
              color: isActive(view) ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: '13px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'color 100ms, border-bottom-color 100ms',
              display: 'flex',
              alignItems: 'center',
              borderRadius: 0,
            }}
            onMouseEnter={(e) => {
              if (!isActive(view))
                (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-primary)';
            }}
            onMouseLeave={(e) => {
              if (!isActive(view))
                (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-secondary)';
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* Theme toggle */}
      <button
        onClick={toggleTheme}
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        style={{
          width: '36px',
          height: '36px',
          background: 'none',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          transition: 'background-color 100ms, color 100ms',
          flexShrink: 0,
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLButtonElement).style.backgroundColor = 'var(--bg-tertiary)';
          (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-primary)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.backgroundColor = 'transparent';
          (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-secondary)';
        }}
      >
        {theme === 'dark' ? (
          <Sun size={18} strokeWidth={1.5} />
        ) : (
          <Moon size={18} strokeWidth={1.5} />
        )}
      </button>
    </header>
  );
}
