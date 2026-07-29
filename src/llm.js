const FLAG_W = 640, FLAG_H = 480
const GW = 24, GH = 16, TW = 6, TH = 4

const SYSTEM_PROMPT =
  'You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.\n' +
  'You are one player in a flag identification game.\n' +
  'Choose exactly one country.\n' +
  'Follow the exact output schema given in the user message.'

export const PROVIDERS = {
  openai: { label: 'OpenAI', placeholder: 'sk-proj-...' },
  anthropic: { label: 'Claude', placeholder: 'sk-ant-...' },
}

export const MODELS = [
  { id: 'gpt-4o',       provider: 'openai', label: 'gpt-4o',       group: 'main', short: '4o',   color: '#5b86c4' },
  { id: 'gpt-5.4',      provider: 'openai', label: 'gpt-5.4',      group: 'main', short: '5.4',  color: '#d4a94b' },
  { id: 'claude-sonnet-4-6', provider: 'anthropic', label: 'claude-sonnet-4-6', group: 'main', short: 's46', color: '#c2683e' },
  { id: 'gpt-4.1-mini', provider: 'openai', label: 'gpt-4.1-mini', group: 'fast', short: '4.1m', color: '#6aa6d4' },
  { id: 'claude-haiku-4-5', provider: 'anthropic', label: 'claude-haiku-4-5', group: 'fast', short: 'h45', color: '#d9a878' },
]

const MODEL_BY_ID = Object.fromEntries(MODELS.map(m => [m.id, m]))

export function anyKey(keys) {
  return !!(keys && (keys.openai || keys.anthropic))
}

export function availableModels(keys) {
  const keyedModels = MODELS.filter(model => keys?.[model.provider])
  return keyedModels.length ? keyedModels : MODELS
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

function dataUrlBase64(url) {
  const comma = url.indexOf(',')
  return comma >= 0 ? url.slice(comma + 1) : url
}

function dataUrlMediaType(url) {
  const match = url.match(/^data:([^;]+)/)
  return match ? match[1] : 'image/png'
}

const ADAPTERS = {
  openai: {
    build(model, key, turns) {
      const messages = [{ role: 'system', content: SYSTEM_PROMPT }]
      for (const turn of turns) {
        if (turn.role === 'user') {
          const content = [{ type: 'text', text: turn.text }]
          if (turn.image) content.push({ type: 'image_url', image_url: { url: turn.image, detail: 'low' } })
          messages.push({ role: 'user', content })
        } else {
          messages.push({ role: 'assistant', content: turn.text })
        }
      }
      const isReasoning = /^gpt-5(\.|-|$)/.test(model)
      const body = { model, messages, max_completion_tokens: isReasoning ? 1500 : 500 }
      if (isReasoning) body.reasoning_effort = 'low'
      if (!isReasoning && model !== 'gpt-4.1-mini') body.response_format = { type: 'json_object' }
      return {
        url: 'https://api.openai.com/v1/chat/completions',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
        body,
      }
    },
    read: data => (data.choices?.[0]?.message?.content || '').trim(),
  },
  anthropic: {
    build(model, key, turns) {
      const messages = turns.map(turn => {
        if (turn.role === 'assistant') return { role: 'assistant', content: turn.text }
        const content = []
        if (turn.image) {
          content.push({
            type: 'image',
            source: {
              type: 'base64',
              media_type: dataUrlMediaType(turn.image),
              data: dataUrlBase64(turn.image),
            },
          })
        }
        content.push({ type: 'text', text: turn.text })
        return { role: 'user', content }
      })
      return {
        url: 'https://api.anthropic.com/v1/messages',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: { model, max_tokens: 500, system: SYSTEM_PROMPT, messages },
      }
    },
    read: data => (data.content || []).filter(part => part.type === 'text').map(part => part.text).join('').trim(),
  },
}

export async function llmInteraction({
  cropDataUrl,
  memoryLines = [],
  model,
  keys,
  m = 3,
  catalog,
  signal,
  maxRetries = 2,
}) {
  const definition = MODEL_BY_ID[model]
  if (!definition) throw new Error(`Unknown model: ${model}`)
  const key = keys?.[definition.provider]
  if (!key) throw new Error(`${PROVIDERS[definition.provider].label} API key required for ${model}.`)
  const adapter = ADAPTERS[definition.provider]

  const text = userPrompt({ memoryLines, m })
  const turns = [{ role: 'user', text, image: cropDataUrl }]

  let lastErr = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const request = adapter.build(model, key, turns)
    const ctl = new AbortController()
    const timer = setTimeout(() => ctl.abort(new Error('Request timed out after 45s')), 45000)
    const onCallerAbort = () => ctl.abort(signal?.reason)
    if (signal) signal.addEventListener('abort', onCallerAbort, { once: true })
    let res
    try {
      res = await fetch(request.url, {
        method: 'POST',
        headers: request.headers,
        signal: ctl.signal,
        body: JSON.stringify(request.body),
      })
    } catch (e) {
      clearTimeout(timer)
      if (signal) signal.removeEventListener('abort', onCallerAbort)
      lastErr = e
      if (attempt >= maxRetries) throw e
      await new Promise(r => setTimeout(r, 300 * (attempt + 1) + Math.random() * 200))
      continue
    }
    clearTimeout(timer)
    if (signal) signal.removeEventListener('abort', onCallerAbort)

    if (!res.ok) {
      const errText = await res.text().catch(() => '')
      const transient = res.status === 429 || res.status >= 500
      if (transient && attempt < maxRetries) {
        lastErr = new Error(`${PROVIDERS[definition.provider].label} ${res.status}: ${errText.slice(0, 200)}`)
        await new Promise(r => setTimeout(r, 400 * Math.pow(2, attempt) + Math.random() * 200))
        continue
      }
      throw new Error(`${PROVIDERS[definition.provider].label} ${res.status}: ${errText.slice(0, 200)}`)
    }
    const data = await res.json()
    const raw = adapter.read(data)

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
