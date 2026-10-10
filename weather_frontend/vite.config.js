import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser calls /api/... on the Vite server, which forwards the request to FastAPI.
// Because the browser only talks to one origin, no CORS setup is needed on the backend.
// To point at another backend:  API_TARGET=http://127.0.0.1:8001 npm run dev
const target = process.env.API_TARGET || "http://127.0.0.1:8000";

const proxy = {
  "/api": {
    target,
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
});
