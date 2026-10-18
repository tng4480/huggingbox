import { useEffect, useRef, useCallback } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';
import { fetchModels, modelMatchesSizeFilter, PIPELINE_OPTIONS, SIZE_OPTIONS, SORT_OPTIONS } from '../../services/huggingfaceApi';
import type { FetchModelsParams } from '../../services/huggingfaceApi';
import ModelCard from './ModelCard';
import SearchBar from './SearchBar';
import FilterDropdown from './FilterDropdown';
import SkeletonCard from '../shared/SkeletonCard';
import Button from '../shared/Button';

const PAGE_SIZE = 24;

export default function BrowseView() {
  const {
    models,
    modelsLoading,
    modelsError,
    modelsPage,
    modelsHasMore,
    searchQuery,
    pipelineFilter,
    sizeFilter,
    sortFilter,
    browseScrollPosition,
    setModels,
    appendModels,
    setModelsPage,
    setModelsHasMore,
    setModelsLoading,
    setModelsError,
    setPipelineFilter,
    setSizeFilter,
    setSortFilter,
    setBrowseScrollPosition,
    settings,
  } = useAppStore();

  const scrollRef = useRef<HTMLDivElement>(null);
  // Holds the AbortController for the current in-flight request
  const abortRef = useRef<AbortController | null>(null);

  // Restore scroll position when returning to browse
  useEffect(() => {
    if (scrollRef.current && browseScrollPosition > 0) {
      scrollRef.current.scrollTop = browseScrollPosition;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadModels = useCallback(
    async (reset: boolean, search: string, pipeline: string | null, page: number, sort?: FetchModelsParams['sort']) => {
      // Cancel any in-flight request immediately
      if (abortRef.current) {
        abortRef.current.abort();
      }
      const controller = new AbortController();
      abortRef.current = controller;

      setModelsLoading(true);
      setModelsError(null);

      try {
        const data = await fetchModels(
          {
            search: search || undefined,
            pipeline_tag: pipeline || undefined,
            page,
            limit: PAGE_SIZE,
            sort: sort ?? sortFilter,
          },
          settings.hfToken || undefined,
          controller.signal
        );

        // If this request was superseded, ignore its result
        if (controller.signal.aborted) return;

        const filtered = sizeFilter
          ? data.filter((m) => modelMatchesSizeFilter(m, sizeFilter))
          : data;

        if (reset) {
          setModels(filtered);
          setModelsPage(0);
        } else {
          // Deduplicate against existing models to prevent HF API pagination overlap
          const existingIds = new Set(useAppStore.getState().models.map((m) => m.modelId ?? m.id));
          appendModels(filtered.filter((m) => !existingIds.has(m.modelId ?? m.id)));
        }
        setModelsHasMore(data.length === PAGE_SIZE);
      } catch (err) {
        if (controller.signal.aborted) return; // expected — ignore
        setModelsError("Couldn't reach Hugging Face. Check your connection and try again.");
      } finally {
        if (!controller.signal.aborted) {
          setModelsLoading(false);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sizeFilter, sortFilter, settings.hfToken]
  );

  // Initial load
  useEffect(() => {
    if (models.length === 0) {
      loadModels(true, searchQuery, pipelineFilter, 0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSearch(q: string) {
    loadModels(true, q, pipelineFilter, 0);
  }

  function handlePipelineChange(val: string) {
    setPipelineFilter(val || null);
    loadModels(true, searchQuery, val || null, 0);
  }

  function handleSizeChange(val: string) {
    setSizeFilter(val || null);
    loadModels(true, searchQuery, pipelineFilter, 0);
  }

  function handleSortChange(val: string) {
    const sort = (val || 'trendingScore') as FetchModelsParams['sort'];
    setSortFilter(sort!);
    loadModels(true, searchQuery, pipelineFilter, 0, sort);
  }

  function handleLoadMore() {
    const nextPage = modelsPage + 1;
    setModelsPage(nextPage);
    loadModels(false, searchQuery, pipelineFilter, nextPage);
  }

  // Infinite scroll
  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    setBrowseScrollPosition(el.scrollTop);
    if (!modelsHasMore || modelsLoading) return;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 200) {
      handleLoadMore();
    }
  }

  const displayModels = sizeFilter
    ? models.filter((m) => modelMatchesSizeFilter(m, sizeFilter))
    : models;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div className="task-shortcuts" aria-label="Choose a task"><span>Start with a task:</span>{[
        ['Text generation', 'text-generation'], ['Transcribe audio', 'automatic-speech-recognition'], ['Generate images', 'text-to-image'],
      ].map(([label, task]) => <Button key={task} onClick={() => handlePipelineChange(task)}>{label}</Button>)}<span>RAM estimates are not runtime verification.</span></div>
      {/* Search + Filters bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-sm)',
          padding: 'var(--space-md) var(--space-xl)',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
          height: '56px',
          backgroundColor: 'var(--bg-primary)',
        }}
      >
        <SearchBar onSearch={handleSearch} />
        <FilterDropdown
          options={PIPELINE_OPTIONS}
          value={pipelineFilter ?? ''}
          onChange={handlePipelineChange}
          placeholder="Pipeline"
        />
        <FilterDropdown
          options={SIZE_OPTIONS}
          value={sizeFilter ?? ''}
          onChange={handleSizeChange}
          placeholder="Size"
        />
        <FilterDropdown
          options={SORT_OPTIONS}
          value={sortFilter}
          onChange={handleSortChange}
          placeholder="Sort"
        />
      </div>

      {/* Scrollable model grid */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-xl)' }}
      >
        {/* Error state */}
        {modelsError && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-sm)',
              backgroundColor: 'rgba(239,68,68,0.1)',
              border: '1px solid var(--error)',
              borderRadius: '6px',
              padding: 'var(--space-md) var(--space-lg)',
              marginBottom: 'var(--space-lg)',
              color: 'var(--error)',
              fontFamily: '"Inter", sans-serif',
              fontSize: '14px',
            }}
          >
            <AlertCircle size={16} strokeWidth={1.5} style={{ flexShrink: 0 }} />
            <span style={{ flex: 1 }}>{modelsError}</span>
            <Button
              variant="secondary"
              icon={<RefreshCw size={14} strokeWidth={1.5} />}
              onClick={() => loadModels(true, searchQuery, pipelineFilter, 0)}
              style={{ borderColor: 'var(--error)', color: 'var(--error)' }}
            >
              Retry
            </Button>
          </div>
        )}

        {/* Loading skeleton (initial / search reset) */}
        {modelsLoading && models.length === 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-lg)' }}>
            {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        )}

        {/* Inline loading indicator when refreshing existing results */}
        {modelsLoading && models.length > 0 && (
          <div
            style={{
              height: '2px',
              background: `linear-gradient(90deg, var(--accent-primary) 0%, transparent 100%)`,
              borderRadius: '1px',
              marginBottom: 'var(--space-lg)',
              animation: 'shimmer 1.2s linear infinite',
              backgroundSize: '400px 100%',
            }}
          />
        )}

        {/* Empty state */}
        {!modelsLoading && !modelsError && displayModels.length === 0 && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 'var(--space-3xl)',
              color: 'var(--text-muted)',
              fontFamily: '"Inter", sans-serif',
              fontSize: '14px',
              gap: 'var(--space-sm)',
            }}
          >
            <p>
              No models found{searchQuery ? ` for "${searchQuery}"` : ''}.
              {searchQuery ? ' Try a different search term.' : ''}
            </p>
          </div>
        )}

        {/* Model grid */}
        {displayModels.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-lg)' }}>
            {displayModels.map((model) => (
              <ModelCard key={model.modelId ?? model.id} model={model} />
            ))}
          </div>
        )}

        {/* Load more / pagination skeletons */}
        {modelsHasMore && displayModels.length > 0 && (
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: 'var(--space-xl)' }}>
            {modelsLoading ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-lg)', width: '100%' }}>
                {Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)}
              </div>
            ) : (
              <Button variant="secondary" onClick={handleLoadMore}>Load More</Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
