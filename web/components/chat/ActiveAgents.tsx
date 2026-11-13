'use client';

import clsx from 'clsx';
import { useEffect, useRef, useState } from 'react';

interface AgentEvent {
  seq: number;
  kind: 'thinking' | 'tool_call' | 'tool_result' | 'error';
  text: string;
}

interface AgentRun {
  run_id: string;
  agent_name: string;
  status: 'running' | 'done' | 'failed';
  elapsed: number;
  events: AgentEvent[];
}

const POLL_MS = 1000;

function useActiveRuns(): AgentRun[] {
  const [runs, setRuns] = useState<AgentRun[]>([]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch('/api/agents/active', { cache: 'no-store' });
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) setRuns(Array.isArray(data?.runs) ? data.runs : []);
        }
      } catch {
        // Panel is best-effort; the chat works without it.
      }
    };
    void tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return runs;
}

export function ActiveAgents() {
  const runs = useActiveRuns();
  if (runs.length === 0) return null;

  return (
    <div className="flex flex-col gap-2 border-t border-gray-200 bg-gray-50 p-3">
      {runs.map(run => (
        <RunCard key={run.run_id} run={run} />
      ))}
    </div>
  );
}

function RunCard({ run }: { run: AgentRun }) {
  const [expanded, setExpanded] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);
  const running = run.status === 'running';

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [run.events.length, expanded]);

  return (
    <div className="rounded-lg border border-gray-200 bg-white text-sm">
      <button
        type="button"
        onClick={() => setExpanded(value => !value)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
      >
        <span
          className={clsx(
            'h-2 w-2 shrink-0 rounded-full',
            running && 'animate-pulse bg-blue-500',
            run.status === 'done' && 'bg-green-500',
            run.status === 'failed' && 'bg-red-500',
          )}
        />
        <span className="flex-1 truncate font-medium text-gray-800">{run.agent_name}</span>
        <span className="text-xs text-gray-500">
          {running ? 'working' : run.status} · {Math.round(run.elapsed)}s
        </span>
      </button>

      {expanded && (
        <div ref={logRef} className="max-h-40 overflow-y-auto border-t border-gray-100 px-3 py-2 font-mono text-xs">
          {run.events.map(event => (
            <div
              key={event.seq}
              className={clsx(
                'whitespace-pre-wrap break-words py-0.5',
                event.kind === 'thinking' && 'italic text-gray-500',
                event.kind === 'tool_call' && 'text-gray-800',
                event.kind === 'tool_result' && 'text-green-700',
                event.kind === 'error' && 'text-red-600',
              )}
            >
              {event.kind === 'tool_call' ? '▸ ' : event.kind === 'tool_result' ? '✓ ' : ''}
              {event.text}
            </div>
          ))}
          {running && <div className="animate-pulse py-0.5 text-gray-400">…</div>}
        </div>
      )}
    </div>
  );
}
