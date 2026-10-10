// Vercel serverless function: the browser calls https://<your-site>/api/chat (https, same site),
// and this function calls your EC2 server over plain http (server to server, so browsers are not involved).
const BACKEND_URL = process.env.BACKEND_URL || "http://13.203.199.191:8000";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Use POST" });
  }
  try {
    const upstream = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.status(upstream.status).json(await upstream.json());
  } catch {
    res.status(502).json({ detail: "The backend is not reachable" });
  }
}
