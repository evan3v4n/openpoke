export const runtime = 'nodejs';

export async function GET() {
  const serverBase = process.env.PY_SERVER_URL || 'http://localhost:8001';
  const url = `${serverBase.replace(/\/$/, '')}/api/v1/agents/active`;

  try {
    const res = await fetch(url, { headers: { Accept: 'application/json' }, cache: 'no-store' });
    const body = await res.text();
    return new Response(body || '{"runs":[]}', {
      status: res.status,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ runs: [], error: error?.message || 'Upstream error' }), {
      status: 502,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  }
}
