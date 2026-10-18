import { useEffect, useRef } from 'react';
import './App.css';
import { useAppStore } from './stores/appStore';
import { useSystemInfo } from './hooks/useSystemInfo';
import { loadSettingsFromFile, saveSettingsToFile } from './services/settingsStorage';
import { useLabStatus } from './hooks/useLabStatus';
import { useExecution } from './hooks/useExecution';
import HeaderBar from './components/layout/HeaderBar';
import StatusBar from './components/layout/StatusBar';
import ErrorBoundary from './components/shared/ErrorBoundary';
import BrowseView from './components/browse/BrowseView';
import ModelDetailView from './components/detail/ModelDetailView';
import MyModelsView from './components/my-models/MyModelsView';
import LabView from './components/lab/LabView';
import SettingsView from './components/settings/SettingsView';

export default function App() {
  const currentView = useAppStore((s) => s.currentView);
  const theme = useAppStore((s) => s.settings.theme);
  const settings = useAppStore((s) => s.settings);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const settingsHydratedRef = useRef(false);

  // Initialise system info polling via Tauri
  useSystemInfo();
  useExecution();
  useLabStatus();

  // Load persisted app settings from app-data on startup
  useEffect(() => {
    let mounted = true;

    void (async () => {
      try {
        const stored = await loadSettingsFromFile();
        if (mounted && stored) {
          updateSettings(stored);
        }
      } catch (error) {
        console.warn('[HuggingBox] Failed to load settings:', error);
      } finally {
        settingsHydratedRef.current = true;
      }
    })();

    return () => {
      mounted = false;
      settingsHydratedRef.current = false;
    };
  }, [updateSettings]);

  // Persist settings whenever user changes them
  useEffect(() => {
    if (!settingsHydratedRef.current) return;

    const timeout = setTimeout(() => {
      void saveSettingsToFile(settings).catch((error) => {
        console.warn('[HuggingBox] Failed to save settings:', error);
      });
    }, 200);

    return () => clearTimeout(timeout);
  }, [settings]);

  // Apply theme to <html> element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <ErrorBoundary>
      <div className="app-shell">
        <HeaderBar />
        <main className="app-main">
          {currentView === 'browse' && <BrowseView />}
          {currentView === 'model-detail' && <ModelDetailView />}
          {currentView === 'my-models' && <MyModelsView />}
          {currentView === 'settings' && <SettingsView />}
          {currentView === 'lab' && <LabView />}
        </main>
        <StatusBar />
      </div>
    </ErrorBoundary>
  );
}
