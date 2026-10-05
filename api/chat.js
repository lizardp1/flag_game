function getOpenAIKey() {
  return (
    process.env.OPENAI_API_KEY ||
    process.env.OPENAI_KEY ||
    process.env.OPENAI_SECRET_KEY ||
    process.env.OPENAI_API_TOKEN ||
    ''
  )
}

function getAnthropicKey() {
  return process.env.ANTHROPIC_API_KEY || ''
}

const MODEL_PROVIDERS = {
  'gpt-4o': 'openai',
  'gpt-5.4': 'openai',
  'gpt-4.1-mini': 'openai',
  'claude-sonnet-4-6': 'anthropic',
  'claude-haiku-4-5': 'anthropic',
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

  const body = await readBody(req)
  let payload
  try {
    payload = JSON.parse(body)
  } catch {
    res.status(400).json({ error: { message: 'Request body must be valid JSON.' } })
    return
  }

  const provider = MODEL_PROVIDERS[payload?.model]
  if (!provider) {
    res.status(400).json({
      error: { message: `Unsupported model: ${payload?.model || 'none'}.` },
    })
    return
  }

  const apiKey = provider === 'anthropic' ? getAnthropicKey() : getOpenAIKey()
  if (!apiKey) {
    const variable = provider === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY'
    res.status(500).json({
      error: {
        message:
          `Server is missing ${variable} on this Vercel deployment.`,
        vercelEnv: process.env.VERCEL_ENV || null,
        gitRef: process.env.VERCEL_GIT_COMMIT_REF || null,
        checked: provider === 'anthropic'
          ? ['ANTHROPIC_API_KEY']
          : ['OPENAI_API_KEY', 'OPENAI_KEY', 'OPENAI_SECRET_KEY', 'OPENAI_API_TOKEN'],
      },
    })
    return
  }

  const upstream = await fetch(
    provider === 'anthropic'
      ? 'https://api.anthropic.com/v1/messages'
      : 'https://api.openai.com/v1/chat/completions',
    {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(provider === 'anthropic'
        ? { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' }
        : { Authorization: `Bearer ${apiKey}` }),
    },
    body,
    },
  )

  res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json')
  res.status(upstream.status).send(await upstream.text())
}
