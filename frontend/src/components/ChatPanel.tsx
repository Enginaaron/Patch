import { useEffect, useRef, useState } from 'react'
import { Mic, MessageCircle, Send, X } from 'lucide-react'
import './ChatPanel.css'
import { speakOmni, stopOmniSpeech } from '../lib/omniSpeaker'

type Role = 'user' | 'assistant'

interface Message {
  role: Role
  content: string
}

interface ChatPanelProps {
  targetText?: string
}

const GREETING: Message = {
  role: 'assistant',
  content: "Hi, I'm Patch. Ask me what I can see, or tell me where to look.",
}

function ChatPanel({ targetText }: ChatPanelProps) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([GREETING])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [mode, setMode] = useState<'live' | 'demo' | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const replyTokenRef = useRef(0)

  useEffect(() => () => { replyTokenRef.current += 1 }, [])

  useEffect(() => {
    fetch('/api/omni/status')
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => setMode(data.mode === 'live' ? 'live' : 'demo'))
      .catch(() => setMode(null))
  }, [])

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, open])

  const send = async () => {
    const text = input.trim()
    if (!text || sending) return

    const nextMessages: Message[] = [...messages, { role: 'user', content: text }]
    setMessages(nextMessages)
    setInput('')
    setSending(true)
    const replyToken = ++replyTokenRef.current

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_text: targetText ?? null,
          messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
        }),
      })
      if (!res.ok) throw new Error('chat failed')
      const data = await res.json()
      if (replyToken !== replyTokenRef.current) return
      setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }])
      speakOmni(data.reply)
      if (data.mode === 'live' || data.mode === 'demo') setMode(data.mode)
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: "Sorry, I couldn't reach my brain just now. Try again?" },
      ])
    } finally {
      setSending(false)
    }
  }

  if (!open) {
    return (
      <button type="button" className="chat-fab" onClick={() => setOpen(true)} aria-label="Ask Patch">
        <MessageCircle size={22} color="white" strokeWidth={2} />
        Ask Patch
      </button>
    )
  }

  return (
    <section className="chat-panel" aria-label="Chat with Patch">
      <header className="chat-panel__header">
        <div className="chat-panel__title">
          <MessageCircle size={18} />
          <span>OMNI Live</span>
          {mode && (
            <span className={`chat-panel__badge chat-panel__badge--${mode}`}>
              {mode === 'live' ? 'Live' : 'Demo'}
            </span>
          )}
        </div>
        <button type="button" className="chat-panel__close" onClick={() => { replyTokenRef.current += 1; stopOmniSpeech(); setOpen(false) }} aria-label="Close chat">
          <X size={18} />
        </button>
      </header>

      <div className="chat-panel__messages" ref={listRef}>
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble chat-bubble--${m.role}`}>
            {m.content}
          </div>
        ))}
        {sending && <div className="chat-bubble chat-bubble--assistant chat-bubble--typing">Patch is looking…</div>}
      </div>

      <form
        className="chat-panel__input-row"
        onSubmit={(e) => {
          e.preventDefault()
          send()
        }}
      >
        <button
          type="button"
          className="chat-icon-button"
          aria-label="Voice input"
          disabled
          title="Voice input isn't available yet"
        >
          <Mic size={18} />
        </button>
        <input
          className="chat-panel__input"
          placeholder="Ask Patch what it sees…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          type="submit"
          className="chat-icon-button chat-icon-button--send"
          aria-label="Send"
          disabled={input.trim().length === 0 || sending}
        >
          <Send size={18} color="white" />
        </button>
      </form>
    </section>
  )
}

export default ChatPanel
