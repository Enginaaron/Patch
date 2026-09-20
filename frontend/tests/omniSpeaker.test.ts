import { test } from 'node:test'
import assert from 'node:assert/strict'
import { enableOmniSpeech, getOmniSpeakerState, muteOmniSpeech, speakOmni, stopOmniSpeech } from '../src/lib/omniSpeaker.ts'

test('OMNI speaker unlocks, plays the latest reply, surfaces errors and cancels stale audio', async () => {
  const originalFetch = globalThis.fetch
  const originalContext = globalThis.AudioContext
  let starts = 0
  let stops = 0
  let fail = false
  let blocked = false
  const texts: string[] = []
  let release: (() => void) | null = null
  class FakeContext {
    state = 'running'
    destination = {}
    async resume() { this.state = blocked ? 'suspended' : 'running' }
    async decodeAudioData() { return {} }
    createBufferSource() {
      return { buffer: null, onended: null, connect() {}, disconnect() {},
        start() { starts++ }, stop() { stops++ } }
    }
  }
  globalThis.AudioContext = FakeContext as unknown as typeof AudioContext
  globalThis.fetch = (async (url, options) => {
    if (url === '/api/speech/synthesize') {
      const { text } = JSON.parse(String(options?.body))
      texts.push(text)
      if (text === 'stale') await new Promise<void>((resolve) => { release = resolve })
      return new Response(JSON.stringify({ audio_url: '/media/speech/test.wav' }), { status: fail ? 502 : 200 })
    }
    return new Response(new Uint8Array([1, 2, 3, 4]))
  }) as typeof fetch
  const settle = () => new Promise((resolve) => setImmediate(resolve))
  try {
    speakOmni('first')
    speakOmni('latest')
    assert.equal(texts.length, 0, 'disabled playback should queue only the latest phrase')
    await enableOmniSpeech()
    await settle()
    assert.deepEqual(texts, ['latest'])
    assert.equal(starts, 1)

    speakOmni('stale')
    speakOmni('new question')
    await settle()
    release?.()
    await settle()
    assert.equal(starts, 2, 'late reply must not play over the new question')
    assert.ok(stops >= 1)

    fail = true
    speakOmni('failed')
    await settle()
    assert.match(getOmniSpeakerState().error ?? '', /OMNI voice is unavailable/)
    assert.equal(getOmniSpeakerState().busy, false)
    fail = false
    blocked = true
    await enableOmniSpeech()
    await settle()
    assert.match(getOmniSpeakerState().error ?? '', /paused by the browser/)
    blocked = false
    await enableOmniSpeech()
    await settle()
    assert.equal(starts, 3, 'retry should replay the failed reply after unlocking')
    assert.equal(getOmniSpeakerState().error, null)
    muteOmniSpeech()
    assert.equal(getOmniSpeakerState().enabled, false)
    assert.equal(getOmniSpeakerState().busy, false)
  } finally {
    stopOmniSpeech()
    globalThis.fetch = originalFetch
    globalThis.AudioContext = originalContext
  }
})
