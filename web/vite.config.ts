import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The orchestrator API runs on :8000; the browser only ever talks to this dev server.
const engine = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: engine, ws: true },
      '/samples': engine,
      '/media': engine,
    },
  },
})
