export const config = { runtime: 'edge' }

export default async function handler(req) {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 })
  }

  if (!process.env.OPENAI_API_KEY) {
    return new Response(
      JSON.stringify({ error: { message: 'Server is missing OPENAI_API_KEY env var.' } }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }

  const body = await req.text()
  const upstream = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body,
  })

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/json' },
  })
}
