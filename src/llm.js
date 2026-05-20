/* Browser helpers for calling OpenAI vision models from the Flag Game.
   API key stays in the browser (localStorage) — never shipped in the bundle. */

const FLAG_W = 640, FLAG_H = 480
const GW = 24, GH = 16, TW = 6, TH = 4

export function rasterizeFlag(svgString) {
  return new Promise((resolve, reject) => {
    const blob = new Blob([svgString], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = FLAG_W; c.height = FLAG_H
      const ctx = c.getContext('2d')
      ctx.drawImage(img, 0, 0, FLAG_W, FLAG_H)
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

export async function llmGuessFlag({ cropDataUrl, candidates, model, apiKey, signal }) {
  const sys = 'You are an agent in a flag identification game. You see a small rectangular crop from a country flag. From the candidate list, identify the most likely country. Reply with ONLY the country name, exactly as it appears in the list. No explanation.'
  const userText = `Candidates: ${candidates.join(', ')}\n\nWhich country flag does this crop most likely come from?`

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    signal,
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: sys },
        {
          role: 'user',
          content: [
            { type: 'text', text: userText },
            { type: 'image_url', image_url: { url: cropDataUrl, detail: 'low' } },
          ],
        },
      ],
      max_tokens: 30,
      temperature: 0,
    }),
  })

  if (!res.ok) {
    const errText = await res.text().catch(() => '')
    throw new Error(`OpenAI ${res.status}: ${errText.slice(0, 200)}`)
  }
  const data = await res.json()
  const raw = (data.choices?.[0]?.message?.content || '').trim()
  const match = candidates.find(c => raw.toLowerCase() === c.toLowerCase())
    || candidates.find(c => raw.toLowerCase().includes(c.toLowerCase()))
  return match || candidates[0]
}
