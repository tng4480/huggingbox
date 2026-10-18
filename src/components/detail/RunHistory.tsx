import { useState } from 'react';
import { useAppStore } from '../../stores/appStore';
import Button from '../shared/Button';

export default function RunHistory() {
  const history = useAppStore(s => s.runHistory);
  const [left, setLeft] = useState('');
  const [right, setRight] = useState('');
  if (!history.length) return null;
  const ids = [left || history[0].id, right || history[1]?.id || history[0].id];
  function download() {
    const blob = new Blob([JSON.stringify(history.filter(r => ids.includes(r.id)), null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'huggingbox-runs.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <details className="run-history"><summary>Saved runs &amp; comparison ({history.length})</summary>
    <p>Last 20 runs, stored on this device. Output previews keep the last 16,000 characters; generated files remain in the workspace.</p>
    <div className="comparison-grid">{ids.map((id, index) => {
      const run = history.find(r => r.id === id) ?? history[0];
      return <section key={index}><label>Compare {index ? 'B' : 'A'}<select value={run.id} onChange={e => (index ? setRight : setLeft)(e.target.value)}>
        {history.map(r => <option key={r.id} value={r.id}>{r.modelId} · {new Date(r.created).toLocaleTimeString()} · {r.state}</option>)}
      </select></label><p>{(run.elapsed/1000).toFixed(1)}s · {run.runtime ?? 'runtime not reported'}{run.tokensPerSecond ? ` · ${run.tokensPerSecond.toFixed(1)} tokens/s` : ''}</p>
      <strong>Input</strong><pre>{run.input}</pre><strong>Output</strong><pre>{run.output}</pre></section>;
    })}</div><Button onClick={download}>Export comparison</Button>
  </details>;
}
