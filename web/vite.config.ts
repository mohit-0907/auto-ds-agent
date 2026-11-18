/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // During development the React app runs on :5173 and FastAPI on :8000.
    // Any request starting with /api is forwarded to FastAPI, so the frontend
    // code can always just call "/api/..." -- in development and in production.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  test: {
    environment: 'jsdom', // a simulated browser, so components can render in tests
    setupFiles: ['./src/test/setup.ts'],
  },
})
