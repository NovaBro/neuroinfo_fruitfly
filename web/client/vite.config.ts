import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// When the API server runs on a different host than Vite (e.g. a separate
// SLURM job), point the proxy at it with VITE_API_TARGET=http://<node>:8000.
const apiTarget = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        timeout: 300_000,
      },
    },
  },
});
