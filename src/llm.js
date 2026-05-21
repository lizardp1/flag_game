/* Browser port of the Flag Game prompts.

   This file is a fork of the upstream `src/llm.js` (which proxies through
   `/api/chat`). Here we call OpenAI directly from the browser with a user-
   supplied API key — no backend needed for the demo. The prompts mirror the
   ones in the paper (`nnd/flag_game/prompts.py`).

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

// Map a UI label ("gpt-4o" / "gpt-5.4") to a real OpenAI model identifier.
// `gpt-5.4` is a cosmetic name in the UI; route it to the small variant so a
// user with an API key still gets a meaningful two-model comparison.
export function realModelName(label) {
  if (label === 'gpt-5.4') return 'gpt-4o-mini'
  return 'gpt-4o'
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

function retryText(errMsg, catalog, m) {
  return (
    `Invalid answer: ${errMsg}\n` +
    `Allowed countries are exactly: ${JSON.stringify(catalog)}\n` +
    'Choose exactly one allowed country from that list. Any other country is invalid.\n' +
    schemaLine(m)
  )
}

function parseResponse(raw, catalog, m) {
  let parsed
  try { parsed = JSON.parse(raw) }
  catch { throw new Error(`Could not parse JSON: ${raw.slice(0, 120)}`) }

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

export async function llmInteraction({
  cropDataUrl,
  memoryLines = [],
  model,            // UI label, e.g. "gpt-4o" or "gpt-5.4"
  apiKey,
  m = 3,
  catalog,
  signal,
  maxRetries = 2,
}) {
  if (!apiKey) throw new Error('OpenAI API key required for LLM mode.')

  const realModel = realModelName(model)
  const text = userPrompt({ memoryLines, m })
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content: [
        { type: 'text', text },
        { type: 'image_url', image_url: { url: cropDataUrl, detail: 'high' } },
      ],
    },
  ]

  let lastErr = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      signal,
      body: JSON.stringify({
        model: realModel,
        messages,
        response_format: { type: 'json_object' },
        max_completion_tokens: m === 1 ? 30 : 120,
      }),
    })

    if (!res.ok) {
      const errText = await res.text().catch(() => '')
      throw new Error(`OpenAI ${res.status}: ${errText.slice(0, 200)}`)
    }
    const data = await res.json()
    const raw = (data.choices?.[0]?.message?.content || '').trim()

    try {
      return parseResponse(raw, catalog, m)
    } catch (e) {
      lastErr = e
      if (attempt >= maxRetries) break
      messages.push({ role: 'assistant', content: raw })
      messages.push({ role: 'user', content: retryText(e.message, catalog, m) })
    }
  }
  throw new Error(`Model failed after ${maxRetries + 1} attempts: ${lastErr?.message || 'unknown'}`)
}
