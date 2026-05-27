/* Browser port of the Flag Game prompts.

   This file is a fork of the upstream `src/llm.js` (which proxies through
   `/api/chat`). Here we call each provider's API directly from the browser
   with a user-supplied key — no backend needed for the demo. Three providers
   are supported (OpenAI, Anthropic, Google); the agent's selected model
   decides which endpoint, key, and request schema are used. The prompts and
   per-provider request shapes mirror the benchmark in `test/benchmark.py`.

   The "Allowed countries" list is omitted from the first prompt — we let the
   model name any country and fuzzy-match against the local catalog when
   displaying results. */

const FLAG_W = 640, FLAG_H = 480
const GW = 24, GH = 16, TW = 6, TH = 4

const SYSTEM_PROMPT =
  'You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.\n' +
  'You are one player in a flag identification game.\n' +
  'Choose exactly one country.\n' +
  'Follow the exact output schema given in the user message.'

/* ─────────── Model registry ───────────
   Each model is {provider, modelId, label, short, color, note, group}.
   `group` is 'main' or 'fast'. `short`/`color` drive the on-grid badge.
   Model selections from test/benchmark.py; deprecation note on sonnet-4-0. */
export const PROVIDERS = {
  openai:    { label: 'OpenAI',    placeholder: 'sk-proj-...' },
  anthropic: { label: 'Anthropic', placeholder: 'sk-ant-...' },
  google:    { label: 'Google',    placeholder: 'AIza...' },
}

export const MODELS = [
  // ── Models ──
  { id: 'gpt-4o',                provider: 'openai',    label: 'gpt-4o',                group: 'main', short: '4o',   color: '#5b86c4' },
  { id: 'gpt-5.4',               provider: 'openai',    label: 'gpt-5.4',               group: 'main', short: '5.4',  color: '#d4a94b' },
  { id: 'gpt-5-mini',            provider: 'openai',    label: 'gpt-5-mini',            group: 'main', short: '5m',   color: '#3f7cb8' },
  { id: 'claude-sonnet-4-6',     provider: 'anthropic', label: 'claude-sonnet-4-6',     group: 'main', short: 's46',  color: '#c2683e' },
  { id: 'claude-sonnet-4-5',     provider: 'anthropic', label: 'claude-sonnet-4-5',     group: 'main', short: 's45',  color: '#cf8a5a' },
  { id: 'gemini-3.5-flash',      provider: 'google',    label: 'gemini-3.5-flash',      group: 'main', short: 'g35',  color: '#5b6fc4' },
  { id: 'gemini-2.5-flash',      provider: 'google',    label: 'gemini-2.5-flash',      group: 'main', short: 'g25',  color: '#7a86d4' },
  // ── Faster ──
  { id: 'gpt-4.1-mini',          provider: 'openai',    label: 'gpt-4.1-mini',          group: 'fast', short: '4.1m', color: '#6aa6d4' },
  { id: 'claude-haiku-4-5',      provider: 'anthropic', label: 'claude-haiku-4-5',      group: 'fast', short: 'h45',  color: '#d9a878' },
  { id: 'claude-sonnet-4-0',     provider: 'anthropic', label: 'claude-sonnet-4-0',     group: 'fast', short: 's40',  color: '#b58a6a' },
  { id: 'gemini-3.1-flash-lite', provider: 'google',    label: 'gemini-3.1-flash-lite', group: 'fast', short: 'g31L', color: '#8a7ac4' },
  { id: 'gemini-2.5-flash-lite', provider: 'google',    label: 'gemini-2.5-flash-lite', group: 'fast', short: 'g25L', color: '#9a8ad0' },
]

const MODEL_BY_ID = Object.fromEntries(MODELS.map(m => [m.id, m]))

export function anyKey(keys) {
  return !!(keys && (keys.openai || keys.anthropic || keys.google))
}

// Models whose provider has a key entered. Hidden (not greyed) when no key.
export function availableModels(keys) {
  if (!keys) return []
  return MODELS.filter(m => keys[m.provider])
}

export function modelMeta(id) {
  return MODEL_BY_ID[id] || { id, label: id, short: id, color: '#5b86c4', provider: 'openai' }
}

// The UI value is the literal API model id.
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

/* Strip ```json …``` fences then parse; fall back to first {...} substring.
   Claude wraps JSON in markdown fences even when asked not to. */
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

/* ─────────── Per-provider request adapters ───────────
   Each `build` turns a provider-neutral turn list into {url, headers, body};
   `read` extracts the assistant text from the parsed response. A turn is
   {role:'user'|'assistant', text, image?} where `image` is a PNG data URL. */
function dataUrlBase64(u) {
  const i = u.indexOf(',')
  return i >= 0 ? u.slice(i + 1) : u
}
function dataUrlMediaType(u) {
  const m = u.match(/^data:([^;]+)/)
  return m ? m[1] : 'image/png'
}

const ADAPTERS = {
  openai: {
    label: 'OpenAI',
    build(model, key, turns) {
      const messages = [{ role: 'system', content: SYSTEM_PROMPT }]
      for (const t of turns) {
        if (t.role === 'user') {
          const content = [{ type: 'text', text: t.text }]
          if (t.image) content.push({ type: 'image_url', image_url: { url: t.image, detail: 'high' } })
          messages.push({ role: 'user', content })
        } else {
          messages.push({ role: 'assistant', content: t.text })
        }
      }
      // gpt-4.1-mini 500s on response_format=json_object + vision; skip it there.
      // extractJson handles unfenced output for that one model.
      const body = { model, messages, max_completion_tokens: 4000 }
      if (model !== 'gpt-4.1-mini') body.response_format = { type: 'json_object' }
      return {
        url: 'https://api.openai.com/v1/chat/completions',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
        body,
      }
    },
    read: data => (data.choices?.[0]?.message?.content || '').trim(),
  },

  anthropic: {
    label: 'Anthropic',
    build(model, key, turns) {
      const messages = turns.map(t => {
        if (t.role === 'user') {
          const content = []
          if (t.image) content.push({ type: 'image', source: { type: 'base64', media_type: dataUrlMediaType(t.image), data: dataUrlBase64(t.image) } })
          content.push({ type: 'text', text: t.text })
          return { role: 'user', content }
        }
        return { role: 'assistant', content: t.text }
      })
      return {
        url: 'https://api.anthropic.com/v1/messages',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          // Required for direct browser calls (otherwise CORS-blocked).
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: { model, max_tokens: 4000, system: SYSTEM_PROMPT, messages },
      }
    },
    read: data => (data.content || []).filter(c => c.type === 'text').map(c => c.text).join('').trim(),
  },

  google: {
    label: 'Google',
    build(model, key, turns) {
      const contents = turns.map(t => {
        const parts = [{ text: t.text }]
        if (t.image) parts.push({ inline_data: { mime_type: dataUrlMediaType(t.image), data: dataUrlBase64(t.image) } })
        return { role: t.role === 'assistant' ? 'model' : 'user', parts }
      })
      return {
        url: `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`,
        headers: { 'Content-Type': 'application/json' },
        body: {
          system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
          contents,
          generationConfig: { response_mime_type: 'application/json', max_output_tokens: 4000 },
        },
      }
    },
    read: data => (data.candidates?.[0]?.content?.parts || []).map(p => p.text || '').join('').trim(),
  },
}

export async function llmInteraction({
  cropDataUrl,
  memoryLines = [],
  model,            // model id, e.g. "gpt-4o", "claude-sonnet-4-6", "gemini-3.5-flash"
  keys,             // {openai, anthropic, google}
  apiKey,           // legacy single-key fallback (treated as OpenAI)
  m = 3,
  catalog,
  signal,
  maxRetries = 2,
}) {
  const def = MODEL_BY_ID[model]
  if (!def) throw new Error(`Unknown model: ${model}`)
  const provider = def.provider
  const key = (keys && keys[provider]) || (provider === 'openai' ? apiKey : null)
  if (!key) throw new Error(`${PROVIDERS[provider].label} API key required for ${model}.`)
  const adapter = ADAPTERS[provider]

  const text = userPrompt({ memoryLines, m })
  const turns = [{ role: 'user', text, image: cropDataUrl }]

  let lastErr = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const { url, headers, body } = adapter.build(model, key, turns)
    const res = await fetch(url, { method: 'POST', headers, signal, body: JSON.stringify(body) })

    if (!res.ok) {
      const errText = await res.text().catch(() => '')
      throw new Error(`${adapter.label} ${res.status}: ${errText.slice(0, 200)}`)
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
