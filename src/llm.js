const FLAG_W = 640, FLAG_H = 480
const GW = 24, GH = 16, TW = 6, TH = 4

const SYSTEM_PROMPT =
  'You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.\n' +
  'You are one player in a flag identification game.\n' +
  'Choose exactly one country.\n' +
  'Follow the exact output schema given in the user message.'

export const PROVIDERS = {
  openai: { label: 'OpenAI' },
  anthropic: { label: 'Anthropic', placeholder: 'sk-ant-...' },
}

export const MODELS = [
  { id: 'gpt-4o',       provider: 'openai', label: 'gpt-4o',       group: 'main', short: '4o',   color: '#5b86c4' },
  { id: 'claude-sonnet-4-6', provider: 'anthropic', label: 'Claude Sonnet 4.6', group: 'main', short: 's46', color: '#c2683e' },
  { id: 'claude-sonnet-4-5', provider: 'anthropic', label: 'Claude Sonnet 4.5', group: 'main', short: 's45', color: '#ab714c' },
  { id: 'claude-haiku-4-5', provider: 'anthropic', label: 'Claude Haiku 4.5', group: 'fast', short: 'h45', color: '#d9a878' },
]

const MODEL_BY_ID = Object.fromEntries(MODELS.map(m => [m.id, m]))

export function anyKey() {
  return true
}

export function availableModels(keys) {
  return MODELS.filter(m => m.provider === 'openai' || !!keys?.anthropic?.trim())
}

export function modelMeta(id) {
  return MODEL_BY_ID[id] || { id, label: id, short: id, color: '#5b86c4', provider: 'openai' }
}

export function realModelName(label) {
  return label
}

function memoryBlock(memoryLines) {
  if (!memoryLines || memoryLines.length === 0) return 'Transcript memory (oldest -> newest): []'
  return 'Transcript memory (oldest -> newest):\n' + memoryLines.map(l => `- ${l}`).join('\n')
}

function schemaLine(m) {
  if (m === 1) return 'Output JSON exactly: {"country":"<one country>"}'
  if (m === 2) return 'Output JSON exactly: {"country":"<one country>","clue":"<short phrase>"}'
  if (m === 3) return 'Output JSON exactly: {"country":"<one country>","reason":"<one sentence>"}'
  throw new Error('m must be 1, 2, or 3')
}

function userPrompt({ memoryLines, m }) {
  return [
    'All players are identifying the same underlying flag.',
    'You always see the same private crop.',
    'Transcript memory shows messages you observed from previous interactions with other players.',
    memoryBlock(memoryLines),
    schemaLine(m),
  ].join('\n')
}

export function rasterizeFlag(svgString) {
  return new Promise((resolve, reject) => {
    const blob = new Blob([svgString], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = FLAG_W; c.height = FLAG_H
      c.getContext('2d').drawImage(img, 0, 0, FLAG_W, FLAG_H)
      URL.revokeObjectURL(url)
      resolve(c)
    }
    img.onerror = e => { URL.revokeObjectURL(url); reject(e) }
    img.src = url
  })
}

export function cropAgentView(flagCanvas, top, left) {
  const cellW = FLAG_W / GW, cellH = FLAG_H / GH
  const sx = left * cellW, sy = top * cellH
  const sw = TW * cellW, sh = TH * cellH
  const c = document.createElement('canvas')
  c.width = Math.round(sw); c.height = Math.round(sh)
  c.getContext('2d').drawImage(flagCanvas, sx, sy, sw, sh, 0, 0, c.width, c.height)
  return c.toDataURL('image/png')
}

function fuzzyMatchCountry(raw, catalog) {
  const norm = s => s.toLowerCase().replace(/[^a-z]/g, '')
  const target = norm(raw)
  const exact = catalog.find(c => norm(c) === target)
  if (exact) return exact
  const contains = catalog.find(c => target.includes(norm(c)) || norm(c).includes(target))
  return contains || null
}

function shuffled(arr) {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function retryText(errMsg, catalog, m) {
  return (
    `Invalid answer: ${errMsg}\n` +
    `Allowed countries are exactly: ${JSON.stringify(shuffled(catalog))}\n` +
    'Choose exactly one allowed country from that list. Any other country is invalid.\n' +
    schemaLine(m)
  )
}

function extractJson(raw) {
  const s = (raw || '').trim()
  if (!s) return null
  let inner = s
  const fence = s.match(/```(?:json)?\s*([\s\S]+?)\s*```/i)
  if (fence) inner = fence[1].trim()
  try { return JSON.parse(inner) } catch { /* try braces below */ }
  const brace = inner.match(/\{[\s\S]*\}/)
  if (brace) { try { return JSON.parse(brace[0]) } catch { /* give up */ } }
  return null
}

function parseResponse(raw, catalog, m) {
  const parsed = extractJson(raw)
  if (!parsed || typeof parsed !== 'object') throw new Error(`Could not parse JSON: ${(raw || '').slice(0, 120)}`)

  const rawCountry = (parsed.country || '').trim()
  if (!rawCountry) throw new Error(`Missing 'country' in response`)
  const matched = catalog ? fuzzyMatchCountry(rawCountry, catalog) : rawCountry
  if (!matched) throw new Error(`'${rawCountry}' is not in the allowed catalog`)

  const clue = typeof parsed.clue === 'string' ? parsed.clue.trim() : null
  const reason = typeof parsed.reason === 'string' ? parsed.reason.trim() : null

  let memoryLine = matched
  if (m === 2 && clue) memoryLine = `${matched} | ${clue}`
  else if (m === 3 && reason) memoryLine = `${matched} | ${reason}`

  return { country: matched, clue, reason, memoryLine }
}

function buildOpenAIRequest(model, turns) {
  const messages = [{ role: 'system', content: SYSTEM_PROMPT }]
  for (const t of turns) {
    if (t.role === 'user') {
      const content = [{ type: 'text', text: t.text }]
      if (t.image) content.push({ type: 'image_url', image_url: { url: t.image, detail: 'low' } })
      messages.push({ role: 'user', content })
    } else {
      messages.push({ role: 'assistant', content: t.text })
    }
  }
  return { model, messages, max_completion_tokens: 500, response_format: { type: 'json_object' } }
}

function buildRequest(model, keys, turns) {
  if (MODEL_BY_ID[model].provider === 'openai') {
    return {
      url: '/api/chat',
      headers: { 'Content-Type': 'application/json' },
      body: buildOpenAIRequest(model, turns),
    }
  }
  // Visitor credentials go directly to Anthropic, never through our server.
  const messages = turns.map(t => {
    if (t.role === 'assistant') return { role: 'assistant', content: t.text }
    const content = []
    if (t.image) {
      const image = t.image.match(/^data:(image\/(?:png|jpeg|gif|webp));base64,(.+)$/)
      if (!image) throw new Error('Claude requires a base64 image crop.')
      content.push({ type: 'image', source: { type: 'base64', media_type: image[1], data: image[2] } })
    }
    content.push({ type: 'text', text: t.text })
    return { role: 'user', content }
  })
  return {
    url: 'https://api.anthropic.com/v1/messages',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': keys.anthropic.trim(),
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: { model, max_tokens: 500, system: SYSTEM_PROMPT, messages },
  }
}

export async function llmInteraction({
  cropDataUrl,
  memoryLines = [],
  model,
  keys,
  m = 3,
  catalog,
  signal = keys?.signal,
  maxRetries = 2,
}) {
  const def = MODEL_BY_ID[model]
  if (!def) throw new Error(`Unknown model: ${model}`)
  if (def.provider === 'anthropic' && !keys?.anthropic?.trim()) {
    throw new Error('Enter your Claude API key to use Claude models.')
  }
  const providerLabel = PROVIDERS[def.provider].label
  signal?.throwIfAborted()

  const text = userPrompt({ memoryLines, m })
  const turns = [{ role: 'user', text, image: cropDataUrl }]

  let lastErr = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    signal?.throwIfAborted()
    const { url, headers, body } = buildRequest(model, keys, turns)
    const ctl = new AbortController()
    const timer = setTimeout(() => ctl.abort(new Error('Request timed out after 45s')), 45000)
    const onCallerAbort = () => ctl.abort(signal?.reason)
    if (signal) signal.addEventListener('abort', onCallerAbort, { once: true })
    let res, data, errText
    try {
      res = await fetch(url, {
        method: 'POST',
        headers,
        signal: ctl.signal,
        body: JSON.stringify(body),
      })
      // Keep timeout and session cancellation active while the body arrives.
      if (res.ok) data = await res.json()
      else errText = await res.text()
    } catch (e) {
      clearTimeout(timer)
      if (signal) signal.removeEventListener('abort', onCallerAbort)
      signal?.throwIfAborted()
      lastErr = e
      if (attempt >= maxRetries) throw e
      await new Promise(r => setTimeout(r, 300 * (attempt + 1) + Math.random() * 200))
      continue
    }
    clearTimeout(timer)
    if (signal) signal.removeEventListener('abort', onCallerAbort)
    signal?.throwIfAborted()

    if (!res.ok) {
      const transient = res.status === 429 || res.status >= 500
      if (transient && attempt < maxRetries) {
        lastErr = new Error(`${providerLabel} ${res.status}: ${errText.slice(0, 200)}`)
        await new Promise(r => setTimeout(r, 400 * Math.pow(2, attempt) + Math.random() * 200))
        continue
      }
      throw new Error(`${providerLabel} ${res.status}: ${errText.slice(0, 200)}`)
    }
    const raw = def.provider === 'anthropic'
      ? (data.content || []).filter(c => c.type === 'text').map(c => c.text).join('').trim()
      : (data.choices?.[0]?.message?.content || '').trim()

    try {
      return parseResponse(raw, catalog, m)
    } catch (e) {
      lastErr = e
      if (attempt >= maxRetries) break
      turns.push({ role: 'assistant', text: raw })
      turns.push({ role: 'user', text: retryText(e.message, catalog, m) })
    }
  }
  throw new Error(`Model failed after ${maxRetries + 1} attempts: ${lastErr?.message || 'unknown'}`)
}
