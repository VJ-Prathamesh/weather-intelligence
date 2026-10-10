const BASE = "http://13.203.199.191:8000"  

export async function sendMessage(message, threadId) {
  let response;
  try {
    response = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running?");
  }

  if (!response.ok) {
    throw new Error(`The server returned an error (${response.status}). Please try again.`);
  }
  return response.json();
}
