import { useCallback, type Dispatch, type SetStateAction } from 'react';
import { useAppStore } from '../stores/appStore';

export function useWorkspaceDraft<T>(modelId: string, key: string, initial: T): [T, Dispatch<SetStateAction<T>>] {
  const saved = useAppStore(s => s.workspaceDrafts[modelId]?.[key]);
  const valid = saved === null ? initial === null : typeof saved === typeof initial || initial === null && typeof saved === 'string';
  const value = (valid && saved !== undefined ? saved : initial) as T;
  const update = useCallback<Dispatch<SetStateAction<T>>>((next) => {
    const current = useAppStore.getState().workspaceDrafts[modelId]?.[key] as T | undefined;
    const resolved = typeof next === 'function' ? (next as (v: T) => T)(current ?? initial) : next;
    useAppStore.getState().setWorkspaceDraft(modelId, {[key]: resolved});
  }, [modelId, key, initial]);
  return [value, update];
}
