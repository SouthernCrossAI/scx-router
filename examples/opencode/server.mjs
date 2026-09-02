#!/usr/bin/env node

import http from 'node:http'
import { createHash } from 'node:crypto'

const PORT = Number(process.env.PORT || 8787)
const UPSTREAM = (process.env.SCX_UPSTREAM_URL || 'https://api.scx.ai/v1').replace(/\/$/, '')
const API_KEY = process.env.SCX_API_KEY

if (!API_KEY) {
  console.error('SCX_API_KEY is not set')
  process.exit(1)
}

const PRESETS = {
  auto:           { quality: 0.65, speed: 0.10, cost: 0.25 },
  'auto-quality': { quality: 0.90, speed: 0.05, cost: 0.05 },
  'auto-fast':    { quality: 0.30, speed: 0.60, cost: 0.10 },
  'auto-cheap':   { quality: 0.15, speed: 0.05, cost: 0.80 },
}
const selectedModels = new Map()

function json(res, status, value) {
  const body = Buffer.from(JSON.stringify(value))
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': body.length,
  })
  res.end(body)
}

async function readJson(req) {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

function conversationKey(body, preset) {
  const messages = Array.isArray(body.messages) ? body.messages : []
  const firstUser = messages.find((message) => message?.role === 'user')
  const seed = JSON.stringify(firstUser?.content ?? messages)
  return `${preset}:${createHash('sha256').update(seed).digest('hex')}`
}

function rememberModel(key, model) {
  if (typeof model !== 'string' || !model || model === 'router') return false
  selectedModels.set(key, model)
  if (selectedModels.size > 500) selectedModels.delete(selectedModels.keys().next().value)
  return true
}

async function relay(upstream, res, key, preset) {
  const contentType = upstream.headers.get('content-type') || 'application/json'
  const headers = { 'content-type': contentType, 'x-scx-router-preset': preset }
  const cacheControl = upstream.headers.get('cache-control')
  if (cacheControl) headers['cache-control'] = cacheControl
  const alreadySelected = selectedModels.has(key)

  if (!contentType.includes('text/event-stream')) {
    const body = Buffer.from(await upstream.arrayBuffer())
    if (upstream.ok && !alreadySelected) {
      try { rememberModel(key, JSON.parse(body.toString('utf8')).model) } catch {}
    }
    headers['content-length'] = body.length
    res.writeHead(upstream.status, headers)
    return res.end(body)
  }

  res.writeHead(upstream.status, headers)
  if (!upstream.body) return res.end()
  const decoder = new TextDecoder()
  let pending = ''
  let remembered = alreadySelected
  for await (const chunk of upstream.body) {
    if (upstream.ok && !remembered) {
      pending += decoder.decode(chunk, { stream: true })
      const lines = pending.split(/\r?\n/)
      pending = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data:')) continue
        const data = line.replace(/^data:\s*/, '')
        if (!data || data === '[DONE]') continue
        try {
          if (rememberModel(key, JSON.parse(data).model)) remembered = true
        } catch {}
      }
    }
    res.write(chunk)
  }
  res.end()
}

async function chat(req, res) {
  let body
  try {
    body = await readJson(req)
  } catch {
    return json(res, 400, { error: { message: 'Invalid JSON body' } })
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return json(res, 400, { error: { message: 'Invalid JSON body' } })
  }

  const preset = Object.hasOwn(PRESETS, body.model) ? body.model : 'auto'
  const routerConfig = Object.hasOwn(body, 'router_config') ? body.router_config : PRESETS[preset]
  const key = conversationKey(body, preset)
  const model = selectedModels.get(key) || 'router'
  const upstreamBody = { ...body, model }
  delete upstreamBody.router_config
  if (model === 'router') upstreamBody.router_config = routerConfig

  const upstream = await fetch(`${UPSTREAM}/chat/completions`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${API_KEY}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify(upstreamBody),
  })
  return relay(upstream, res, key, preset)
}

const server = http.createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname
  try {
    if (req.method === 'POST' && pathname.endsWith('/chat/completions')) {
      return await chat(req, res)
    }
    if (req.method === 'GET' && pathname.endsWith('/models')) {
      return json(res, 200, {
        object: 'list',
        data: Object.keys(PRESETS).map((id) => ({ id, object: 'model', owned_by: 'scx' })),
      })
    }
    return json(res, 404, { error: { message: 'Not found' } })
  } catch (error) {
    console.error(error)
    if (!res.headersSent) return json(res, 502, { error: { message: 'SCX upstream unavailable' } })
    res.end()
  }
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`SCX router shim listening on http://127.0.0.1:${PORT}/v1`)
})
