# Website model access

The website offers GPT-4o by default. The browser sends GPT-4o requests to
`/api/chat`; that Vercel function reads `OPENAI_API_KEY` from the deployment's
server environment. Existing alternative environment names remain supported.
Never put the key in a `VITE_` variable or in browser code. No visitor OpenAI key
or app password is required by this version.

The server accepts only `gpt-4o`, fixes the response format to JSON, and caps each
response at 500 completion tokens. It does not forward extra client options such
as `n`, `stream`, or larger token budgets. Provider errors are sanitized so an
authentication error cannot expose the hosted credential.

Visitors can optionally enter their own Anthropic API key and select **Enable
Claude**. This unlocks the tested Claude Sonnet 4.6 (`claude-sonnet-4-6`), Sonnet
4.5 (`claude-sonnet-4-5`), and Haiku 4.5 (`claude-haiku-4-5`) models in all three
game modes. Claude requests go directly from the browser to Anthropic's Messages
API and are billed to the visitor's Anthropic account. Keys live only in React
memory; the app does not write them to browser storage or send them to Vercel.
Applying, replacing, or clearing a key aborts pending browser requests and resets
all games, including their model selections. Reloading also clears the key.
An already accepted provider request may still incur usage after browser abort.

The older tested Sonnet 4 model is omitted because Anthropic retired it on
June 15, 2026. Sonnet 4.5 remains callable but is scheduled to retire on
November 30, 2026; remove it from the registry when it retires. See the
[Anthropic lifecycle documentation](https://platform.claude.com/docs/en/about-claude/model-deprecations).
These choices preserve the tested model set rather than adding newer models.

Run the offline regression checks with `node --test tests/web-model-access.test.js`
and the production build with `npm run build`. Tests replace network requests
and use dummy credentials; they do not make paid calls. `npm run dev` previews
the frontend; hosted GPT-4o requires the Vercel function and deployment secret.
This commit does not change the deployment's environment variables or publish it.
