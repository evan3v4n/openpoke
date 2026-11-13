'use client';

import { useEffect, useState } from 'react';

interface Draft {
  agent_name: string;
  recipient: string;
  subject: string;
}

interface Reminder {
  id: number;
  agent_name: string;
  payload: string;
  next_trigger: string | null;
  recurrence_rule: string | null;
  timezone: string | null;
}

interface Pending {
  drafts: Draft[];
  reminders: Reminder[];
}

const POLL_MS = 5000;

function usePending(): Pending {
  const [pending, setPending] = useState<Pending>({ drafts: [], reminders: [] });

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch('/api/agents/pending', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          setPending({
            drafts: Array.isArray(data?.drafts) ? data.drafts : [],
            reminders: Array.isArray(data?.reminders) ? data.reminders : [],
          });
        }
      } catch {
        // Best-effort panel.
      }
    };
    void tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return pending;
}

function formatWhen(iso: string | null, timezone: string | null): string {
  if (!iso) return 'no next time';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  try {
    return date.toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      ...(timezone ? { timeZone: timezone } : {}),
    });
  } catch {
    return date.toLocaleString();
  }
}

export function PendingItems() {
  const { drafts, reminders } = usePending();
  const total = drafts.length + reminders.length;

  return (
    <aside className="card flex flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-gray-800">Pending &amp; recurring</h2>
        <span className="chip">{total}</span>
      </div>

      <div className="flex max-h-72 flex-col gap-3 overflow-y-auto p-3 text-sm">
        {total === 0 && (
          <p className="px-1 py-4 text-center text-gray-500">
            Nothing waiting. Unsent drafts and reminders show up here.
          </p>
        )}

        {drafts.length > 0 && (
          <Section title="Unsent drafts">
            {drafts.map((draft, index) => (
              <Item
                key={`draft-${index}`}
                title={draft.subject || '(no subject)'}
                detail={draft.recipient ? `To ${draft.recipient}` : 'No recipient'}
                agent={draft.agent_name}
              />
            ))}
          </Section>
        )}

        {reminders.length > 0 && (
          <Section title="Reminders">
            {reminders.map(reminder => (
              <Item
                key={`reminder-${reminder.id}`}
                title={reminder.payload}
                detail={`${formatWhen(reminder.next_trigger, reminder.timezone)}${
                  reminder.recurrence_rule ? ' · repeats' : ''
                }`}
                agent={reminder.agent_name}
                recurring={Boolean(reminder.recurrence_rule)}
              />
            ))}
          </Section>
        )}
      </div>
    </aside>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="px-1 text-xs font-medium uppercase tracking-wide text-gray-500">{title}</div>
      {children}
    </div>
  );
}

function Item({
  title,
  detail,
  agent,
  recurring,
}: {
  title: string;
  detail: string;
  agent: string;
  recurring?: boolean;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-3 py-2">
      <div className="line-clamp-2 break-words text-gray-800">
        {recurring && <span aria-hidden>↻ </span>}
        {title}
      </div>
      <div className="mt-0.5 text-xs text-gray-500">{detail}</div>
      <div className="mt-0.5 truncate text-xs text-gray-400">{agent}</div>
    </div>
  );
}
