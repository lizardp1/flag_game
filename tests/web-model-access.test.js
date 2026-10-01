import test, { afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { Readable } from 'node:stream'
import handler from '../api/chat.js'
import { MODELS, availableModels, llmInteraction } from '../src/llm.js'

const originalFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = originalFetch })

const result = { country: 'Japan', reason: 'A red circle on white.' }
const cropDataUrl = 'data:image/png;base64,dGVzdA=='
const turn = { model: 'gpt-4o', cropDataUrl, catalog: ['Japan', 'France'], maxRetries: 0 }
const openaiResponse = () => Response.json({ choices: [{ message: { content: JSON.stringify(result) } }] })
const claudeResponse = () => Response.json({ content: [{ type: 'text', text: '```json\n' + JSON.stringify(result) + '\n```' }] })

// Only fake credentials are assigned in this isolated test process.
for (const name of ['OPENAI_API_KEY', 'OPENAI_KEY', 'OPENAI_SECRET_KEY', 'OPENAI_API_TOKEN']) delete process.env[name]
process.env.OPENAI_API_KEY = 'test-only-server-credential'

async function callHandler(body, method = 'POST', streamed = false) {
  const req = streamed ? Readable.from([JSON.stringify(body)]) : { body }
  req.method = method
  const res = {
    headers: {},
    setHeader(name, value) { this.headers[name] = value },
    status(status) { this.statusCode = status; return this },
    json(value) { this.body = value; return this },
    send(value) { this.body = value; return this },
  }
  await handler(req, res)
  return res
}

test('only GPT-4o is available without a Claude key, including old browser key shapes', () => {
  for (const keys of [undefined, {}, { anthropic: '  ' }, { openai: 'legacy', google: 'legacy' }]) {
    assert.deepEqual(availableModels(keys).map(m => m.id), ['gpt-4o'])
  }
  assert.deepEqual(availableModels({ anthropic: 'test-only-visitor-credential' }).map(m => m.id), [
    'gpt-4o', 'claude-sonnet-4-6', 'claude-sonnet-4-5', 'claude-haiku-4-5',
  ])
  assert.equal(MODELS.filter(m => m.provider === 'openai').length, 1)
})

test('GPT-4o uses the same-origin proxy without sending any visitor credential', async () => {
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/chat')
    assert.deepEqual(options.headers, { 'Content-Type': 'application/json' })
    assert.ok(!options.body.includes('test-only-visitor-credential'))
    const body = JSON.parse(options.body)
    assert.equal(body.model, 'gpt-4o')
    assert.equal(body.messages[1].content[1].image_url.url, cropDataUrl)
    return openaiResponse()
  }
  const response = await llmInteraction({ ...turn, keys: { anthropic: 'test-only-visitor-credential' } })
  assert.equal(response.memoryLine, 'Japan | A red circle on white.')
})

test('each Claude model sends only the visitor key directly to Anthropic', async () => {
  for (const model of MODELS.filter(m => m.provider === 'anthropic')) {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, 'https://api.anthropic.com/v1/messages')
      assert.equal(options.headers['x-api-key'], 'test-only-visitor-credential')
      assert.equal(options.headers['anthropic-version'], '2023-06-01')
      assert.equal(options.headers['anthropic-dangerous-direct-browser-access'], 'true')
      assert.equal(options.headers.Authorization, undefined)
      const body = JSON.parse(options.body)
      assert.equal(body.model, model.id)
      assert.equal(body.max_tokens, 500)
      assert.equal(body.messages[0].content[0].source.data, 'dGVzdA==')
      assert.match(body.system, /valid JSON/)
      assert.ok(!options.body.includes('test-only-visitor-credential'))
      return claudeResponse()
    }
    assert.equal((await llmInteraction({ ...turn, model: model.id, keys: { anthropic: ' test-only-visitor-credential ' } })).country, 'Japan')
  }
})

test('disabled models and missing Claude credentials fail before a network call', async () => {
  globalThis.fetch = () => { assert.fail('Unexpected network request') }
  for (const model of ['gpt-5.4', 'gpt-4.1-mini', 'gemini-2.5-flash']) {
    await assert.rejects(llmInteraction({ ...turn, model }), /Unknown model/)
  }
  await assert.rejects(llmInteraction({ ...turn, model: 'claude-sonnet-4-6' }), /Claude API key/)
})

test('Claude format retries retain the image, memory, and correction transcript', async () => {
  let calls = 0
  globalThis.fetch = async (url, options) => {
    assert.equal(url, 'https://api.anthropic.com/v1/messages')
    const body = JSON.parse(options.body)
    assert.match(body.messages[0].content[1].text, /France \| Blue stripe/)
    if (++calls === 1) return Response.json({ content: [{ type: 'text', text: 'not JSON' }] })
    assert.equal(body.messages[1].role, 'assistant')
    assert.equal(body.messages[1].content, 'not JSON')
    assert.match(body.messages[2].content[0].text, /Allowed countries are exactly/)
    return claudeResponse()
  }
  await llmInteraction({ ...turn, model: 'claude-haiku-4-5', keys: { anthropic: 'test-only-visitor-credential' }, memoryLines: ['France | Blue stripe'], maxRetries: 1 })
  assert.equal(calls, 2)
})

test('invalid Claude credentials surface without falling back to the hosted key', async () => {
  let calls = 0
  globalThis.fetch = async url => {
    assert.equal(url, 'https://api.anthropic.com/v1/messages')
    calls++
    return Response.json({ error: { message: 'Invalid API key' } }, { status: 401 })
  }
  await assert.rejects(llmInteraction({ ...turn, model: 'claude-sonnet-4-6', keys: { anthropic: 'test-only-visitor-credential' }, maxRetries: 2 }), /Anthropic 401/)
  assert.equal(calls, 1)
})

test('clearing a credential session cancels its request and prevents retries', async () => {
  const controller = new AbortController()
  let calls = 0
  globalThis.fetch = async (url, { signal }) => {
    calls++
    return new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), { once: true })
      controller.abort()
    })
  }
  await assert.rejects(llmInteraction({ ...turn, keys: { signal: controller.signal }, maxRetries: 2 }), { name: 'AbortError' })
  await assert.rejects(llmInteraction({ ...turn, keys: { signal: controller.signal } }), { name: 'AbortError' })
  assert.equal(calls, 1)
})

test('session cancellation also covers a pending response body', async () => {
  const controller = new AbortController()
  let calls = 0
  globalThis.fetch = async (url, { signal }) => {
    calls++
    return { ok: true, json: () => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), { once: true })
      controller.abort()
    }) }
  }
  await assert.rejects(llmInteraction({ ...turn, keys: { signal: controller.signal }, maxRetries: 2 }), { name: 'AbortError' })
  assert.equal(calls, 1)
})

test('server accepts parsed and streamed JSON, fixes model/budget, and uses only its own key', async () => {
  for (const streamed of [false, true]) {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, 'https://api.openai.com/v1/chat/completions')
      assert.equal(options.headers.Authorization, 'Bearer test-only-server-credential')
      assert.deepEqual(JSON.parse(options.body), {
        model: 'gpt-4o', messages: [{ role: 'user', content: 'Return JSON.' }],
        max_completion_tokens: 500, response_format: { type: 'json_object' },
      })
      return openaiResponse()
    }
    const response = await callHandler({ model: 'gpt-4o', messages: [{ role: 'user', content: 'Return JSON.' }], n: 100, stream: true, max_completion_tokens: 100000, apiKey: 'untrusted' }, 'POST', streamed)
    assert.equal(response.statusCode, 200)
    assert.equal(response.headers['Cache-Control'], 'no-store')
  }
})

test('server rejects malformed bodies and every non-GPT-4o model before forwarding', async () => {
  globalThis.fetch = () => { assert.fail('Unexpected network request') }
  for (const body of ['invalid json', 'null', '{}', { model: 'gpt-5.4' }, { model: 'claude-sonnet-4-6' }, { model: 'gpt-4o', messages: [] }]) {
    assert.equal((await callHandler(body)).statusCode, 400)
  }
  assert.equal((await callHandler({}, 'GET')).statusCode, 405)
})

test('server redacts upstream errors and handles connection failure', async () => {
  const body = { model: 'gpt-4o', messages: [{ role: 'user', content: 'Return JSON.' }] }
  globalThis.fetch = async () => Response.json({ error: { message: 'Incorrect key: test-only-server-credential' } }, { status: 401 })
  const rejected = await callHandler(body)
  assert.equal(rejected.statusCode, 401)
  assert.ok(!JSON.stringify(rejected.body).includes('test-only-server-credential'))
  globalThis.fetch = async () => { throw new Error('Network failed') }
  assert.equal((await callHandler(body)).statusCode, 502)
})
