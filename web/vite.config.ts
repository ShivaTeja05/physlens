import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

// The orchestrator API runs on :8000; the browser only ever talks to this dev server.
const engine = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  // Relative asset paths, so the built simulator page also works from a static folder (a shared copy).
  base: './',
  build: {
    rollupOptions: { input: { main: resolve(__dirname, 'index.html'), sim: resolve(__dirname, 'sim.html') } },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: engine, ws: true },
      '/samples': engine,
      '/media': engine,
    },
  },
})
