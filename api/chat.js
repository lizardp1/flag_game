function getOpenAIKey() {
  return (
    process.env.OPENAI_API_KEY ||
    process.env.OPENAI_KEY ||
    process.env.OPENAI_SECRET_KEY ||
    process.env.OPENAI_API_TOKEN ||
    ''
  )
}

async function readBody(req) {
  if (req.body) {
    return typeof req.body === 'string' ? req.body : JSON.stringify(req.body)
  }

  const chunks = []
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  return Buffer.concat(chunks).toString('utf8')
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).send('Method Not Allowed')
    return
  }

  const apiKey = getOpenAIKey()
  if (!apiKey) {
    res.status(500).json({
      error: {
        message:
          'Server is missing an OpenAI API key env var. Expected OPENAI_API_KEY on this Vercel deployment.',
        vercelEnv: process.env.VERCEL_ENV || null,
        gitRef: process.env.VERCEL_GIT_COMMIT_REF || null,
        checked: ['OPENAI_API_KEY', 'OPENAI_KEY', 'OPENAI_SECRET_KEY', 'OPENAI_API_TOKEN'],
      },
    })
    return
  }

  const body = await readBody(req)
  const upstream = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body,
  })

  res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json')
  res.status(upstream.status).send(await upstream.text())
}
