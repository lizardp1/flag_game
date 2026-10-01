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
  res.setHeader('Cache-Control', 'no-store')
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST')
    res.status(405).send('Method Not Allowed')
    return
  }

  let body
  try {
    body = JSON.parse(await readBody(req))
  } catch {
    res.status(400).json({ error: { message: 'Request body must be valid JSON.' } })
    return
  }
  if (body?.model !== 'gpt-4o') {
    res.status(400).json({ error: { message: 'Only gpt-4o is available through this server.' } })
    return
  }
  if (!Array.isArray(body.messages) || !body.messages.length) {
    res.status(400).json({ error: { message: 'A non-empty messages array is required.' } })
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

  try {
    // Do not let browser-supplied model, token limits, n, or other options
    // expand the hosted access beyond the game's single GPT-4o response.
    const upstream = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        messages: body.messages,
        max_completion_tokens: 500,
        response_format: { type: 'json_object' },
      }),
    })

    if (!upstream.ok) {
      // Provider authentication errors can echo parts of the server key.
      const message = upstream.status === 429
        ? 'GPT-4o is busy or its usage limit has been reached. Please try again later.'
        : 'The hosted GPT-4o request failed. Please try again or contact the site owner.'
      res.status(upstream.status).json({ error: { message } })
      return
    }
    res.setHeader('Content-Type', 'application/json')
    res.status(upstream.status).send(await upstream.text())
  } catch {
    res.status(502).json({ error: { message: 'Could not reach hosted GPT-4o. Please try again.' } })
  }
}
