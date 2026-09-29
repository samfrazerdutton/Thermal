import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// THERMAL_API_PORT lets a developer run the API on a non-default port (e.g.
// because something else already occupies 8000) without editing this file.
const apiPort = process.env.THERMAL_API_PORT ?? '8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': `http://127.0.0.1:${apiPort}`,
    },
  },
})
