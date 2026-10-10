import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { sendMessage } from "./api";

const SUGGESTIONS = [
  "How hot was Delhi in May 2024?",
  "Will it rain tomorrow in Jalgaon?",
  "What is the maximum temperature tomorrow in Pune?",
];

// crypto.randomUUID only exists on secure pages (https or localhost), so keep a fallback
const newId = () =>
  crypto.randomUUID?.() ?? `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;

// Small helpers: keep the chat in localStorage so a page refresh does not lose it
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage not available (private window): the chat still works, it just is not remembered */
  }
}

export default function App() {
  // threadId is what the backend uses to remember this conversation
  const [threadId, setThreadId] = useState(() => load("weather.threadId", null) ?? newId());
  const [messages, setMessages] = useState(() => load("weather.messages", []));
  // true while the agent has asked a question and is waiting for the user's answer
  const [waiting, setWaiting] = useState(() => load("weather.waiting", false));
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => save("weather.threadId", threadId), [threadId]);
  useEffect(() => save("weather.messages", messages), [messages]);
  useEffect(() => save("weather.waiting", waiting), [waiting]);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const addMessage = (message) =>
    setMessages((previous) => [...previous, { id: newId(), ...message }]);

  async function send(text) {
    const message = text.trim();
    if (!message || loading) return;

    setInput("");
    addMessage({ role: "user", text: message });
    setLoading(true);

    try {
      const reply = await sendMessage(message, threadId);
      if (reply.status === "needs_input") {
        // the agent needs one more detail: show its question; the next message is the answer
        addMessage({ role: "assistant", kind: "question", text: reply.question });
        setWaiting(true);
      } else {
        addMessage({ role: "assistant", text: reply.answer });
        setWaiting(false);
      }
    } catch (error) {
      addMessage({ role: "assistant", kind: "error", text: error.message });
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setThreadId(newId()); // a new id = a new conversation with an empty memory on the backend
    setMessages([]);
    setWaiting(false);
    setInput("");
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(input);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Weather Assistant</h1>
          <p>History for Delhi, Pune and Kolkata · forecasts for any city</p>
        </div>
        <button className="ghost" onClick={newChat} disabled={loading}>
          New chat
        </button>
      </header>

      <main className="messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about past weather or the forecast. Try one of these:</p>
            <div className="chips">
              {SUGGESTIONS.map((suggestion) => (
                <button key={suggestion} className="chip" onClick={() => send(suggestion)}>
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`row ${message.role}`}>
            <div className={`bubble ${message.role} ${message.kind ?? ""}`}>
              {message.kind === "question" && <span className="label">One detail needed</span>}
              {message.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
              ) : (
                message.text
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="row assistant">
            <div className="bubble assistant typing" aria-label="The assistant is typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          rows={1}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={waiting ? "Type your answer…" : "Ask about the weather…"}
          disabled={loading}
          aria-label="Message"
        />
        <button onClick={() => send(input)} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>
      <p className="hint">Enter to send · Shift+Enter for a new line</p>
    </div>
  );
}
