import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
export function useLabStatus() {
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let mounted = true;
    let pending = false;
    async function refresh() {
      if (pending) return;
      pending = true;
      try {
        const runs =
          await invoke<{ status: string; action: string }[]>("lab_runs");
        if (mounted) {
          const active = runs.find((r) => r.status === "running");
          useAppStore.setState({
            labJobLabel: active ? `Lab: ${active.action}…` : null,
          });
        }
      } catch {
        /* Lab view displays detailed errors. */
      } finally {
        pending = false;
      }
    }
    void refresh();
    const timer = setInterval(refresh, 1500);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);
}
