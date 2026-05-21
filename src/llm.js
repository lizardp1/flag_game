/* Browser port of the Flag Game prompts from nnd/flag_game/prompts.py.
   The "Allowed countries" line is omitted — we let the model name any country
   and fuzzy-match against the local catalog when displaying results. */

const FLAG_W = 640, FLAG_H = 480
const GW = 24, GH = 16, TW = 6, TH = 4

const SYSTEM_PROMPT =
  'You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.\n' +
  'You are one player in a flag identification game.\n' +
  'Choose exactly one country.\n' +
  'Follow the exact output schema given in the user message.'

function memoryBlock(memoryLines) {
  if (!memoryLines || memoryLines.length === 0) return 'Transcript memory (oldest -> newest): []'
  return 'Transcript memory (oldest -> newest):\n' + memoryLines.map(l => `- ${l}`).join('\n')
}

function susceptibilityLine(a) {
  let g
  if (a <= 0.2) g = 'Rely mostly on your own crop and treat transcript memory as weak evidence.'
  else if (a <= 0.4) g = 'Give somewhat more weight to your own crop than to transcript memory.'
  else if (a <= 0.6) g = 'Balance your own crop and transcript memory.'
  else if (a <= 0.8) g = 'Give somewhat more weight to transcript memory than to your own crop.'
  else g = 'Treat transcript memory as strong evidence and update readily toward it.'
  return `Social susceptibility a = ${a.toFixed(2)}. ${g}`
}

function schemaLine(m) {
  if (m === 1) return 'Output JSON exactly: {"country":"<one country>"}'
  if (m === 2) return 'Output JSON exactly: {"country":"<one country>","clue":"<short phrase>"}'
  if (m === 3) return 'Output JSON exactly: {"country":"<one country>","reason":"<one sentence>"}'
  throw new Error('m must be 1, 2, or 3')
}

function userPrompt({ memoryLines, m, socialSusceptibility, promptSocialSusceptibility }) {
  const parts = [
    'All players are identifying the same underlying flag.\n' +
    'You always see the same private crop.\n' +
    'Transcript memory shows messages you observed from previous interactions with other players.\n' +
    memoryBlock(memoryLines),
  ]
  if (promptSocialSusceptibility) parts.push(susceptibilityLine(socialSusceptibility))
  parts.push(schemaLine(m))
  return parts.join('\n')
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
  let exact = catalog.find(c => norm(c) === target)
  if (exact) return exact
  let contains = catalog.find(c => target.includes(norm(c)) || norm(c).includes(target))
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
  model,
  apiKey,
  m = 3,
  socialSusceptibility = 0.5,
  promptSocialSusceptibility = false,
  catalog,
  signal,
  maxRetries = 2,
}) {
  const text = userPrompt({ memoryLines, m, socialSusceptibility, promptSocialSusceptibility })
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
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      signal,
      body: JSON.stringify({
        model,
        messages,
        response_format: { type: 'json_object' },
        max_completion_tokens: 4000,
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
