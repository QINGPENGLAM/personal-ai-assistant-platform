import { defineConfig } from 'vitest/config';

const backendPort = Number(process.env.QPL_BROWSER_BACKEND_PORT ?? 8000);
if (!Number.isInteger(backendPort) || backendPort < 1024 || backendPort > 65535) throw new Error('Invalid loopback backend port.');
export default defineConfig({
  preview: {
    host: '127.0.0.1', strictPort: true,
    proxy: { '/api': `http://127.0.0.1:${backendPort}`, '/mcp': `http://127.0.0.1:${backendPort}` },
  },
  server: {
    host: '127.0.0.1', port: 5173, strictPort: true,
    proxy: { '/api': `http://127.0.0.1:${backendPort}`, '/mcp': `http://127.0.0.1:${backendPort}` },
  },
  test: { environment: 'jsdom', include: ['src/**/*.test.ts'] },
});
