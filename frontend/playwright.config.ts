import { defineConfig } from '@playwright/test';
const frontendPort = Number(process.env.QPL_BROWSER_FRONTEND_PORT ?? 5174);
const backendPort = Number(process.env.QPL_BROWSER_BACKEND_PORT ?? 8001);
for (const port of [frontendPort, backendPort]) if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid test port.');
export default defineConfig({
  testDir: './e2e', testMatch: process.env.RUN_REAL_AGENT_TESTS === '1' ? 'real-lifecycle.spec.ts' : 'core.spec.ts', workers: 1,
  projects: [
    { name: 'desktop', use: { viewport: { width: 1280, height: 900 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 } } },
  ],
  use: { baseURL: `http://127.0.0.1:${frontendPort}`, trace: 'retain-on-failure' },
  webServer: [
    { command: `../.venv/bin/python ../scripts/serve-test.py --mode ${process.env.RUN_REAL_AGENT_TESTS === '1' ? 'real' : 'fake'} --port ${backendPort} --frontend-port ${frontendPort}`, url: `http://127.0.0.1:${backendPort}/health`, reuseExistingServer: false },
    { command: process.env.RUN_REAL_AGENT_TESTS === '1' ? `npm run build && npm exec vite preview -- --host 127.0.0.1 --port ${frontendPort} --strictPort` : `npm run dev -- --port ${frontendPort}`, env: { QPL_BROWSER_BACKEND_PORT: String(backendPort) }, url: `http://127.0.0.1:${frontendPort}`, reuseExistingServer: false },
  ],
});
