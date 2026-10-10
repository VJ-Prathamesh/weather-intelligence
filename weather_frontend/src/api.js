// One function that talks to the FastAPI backend.
// The Vite dev server forwards /api/chat to http://127.0.0.1:8000/chat (see vite.config.js).
//
// The backend answers with one of:
//   { status: "done",        answer: "..." }     the final answer
//   { status: "needs_input", question: "..." }   the agent needs one more detail from the user
export async function sendMessage(message, threadId) {
  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running on port 8000?");
  }

  if (!response.ok) {
    throw new Error(`The server returned an error (${response.status}). Please try again.`);
  }
  return response.json();
}
