import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { constants, openSync, fstatSync, readFileSync, closeSync } from 'node:fs';

// Real execution is opt-in and only valid on the separately provisioned test host.
// Never infer ownership from an IP address or a caller-supplied container name.
function independentBackend() {
  if (process.env.RUN_REAL_AGENT_TESTS !== '1') throw new Error('Real acceptance requires explicit opt-in.');
  const fd = openSync('/etc/qpl-acceptance-host.json', constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.uid !== 0 || (stat.mode & 0o022) || stat.size > 4096) throw new Error('Invalid independent host marker.');
    const marker = JSON.parse(readFileSync(fd, 'utf8'));
    if (Object.keys(marker).sort().join(',') !== 'project,purpose,session_id' || marker.project !== 'hackathon-2-qpl' || marker.purpose !== 'isolated-acceptance' || !/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(marker.session_id)) throw new Error('Invalid independent host identity.');
    return { session: marker.session_id as string, name: `qpl-real-backend-${marker.session_id}` };
  } finally { closeSync(fd); }
}
function docker(...args: string[]) {
  const env = { ...process.env };
  for (const key of ['DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH']) delete env[key];
  return execFileSync('docker', ['--host', 'unix:///var/run/docker.sock', ...args], { env, encoding: 'utf8', timeout: 600_000, maxBuffer: 1024 * 1024 });
}
function ownedBackend() {
  const owned = independentBackend();
  const [info] = JSON.parse(docker('inspect', owned.name));
  expect(info.Name).toBe(`/${owned.name}`);
  expect(info.Config.Labels['qpl.test-session']).toBe(owned.session);
  expect(info.Config.Labels['qpl.purpose']).toBe('real-test-backend');
  expect(info.Config.Labels['qpl.platform']).toBe('personal-assistant');
  expect(info.Config.User).toBe('1000:1000');
  expect(info.HostConfig.Privileged).toBe(false);
  expect(info.State.Running).toBe(true);
  expect(info.Image).toMatch(/^sha256:[0-9a-f]{64}$/);
  return owned;
}
async function file(request: APIRequestContext, base: string, path: string) {
  const response = await request.get(`${base}/files/preview?path=${encodeURIComponent(path)}`);
  if (response.status() === 404) return null;
  expect(response.ok(), `File read failed: ${response.status()}`).toBe(true);
  return (await response.json()).text as string | null;
}
async function send(page: Page, prompt: string) {
  await page.getByLabel('Message your assistant').fill(prompt);
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
}
function commandPrompt(start: string, finish: string, result: string, delay: number) {
  return `Run exactly one Python shell command now: import pathlib,time; pathlib.Path('/workspace/${start}').write_text('STARTED'); time.sleep(${delay}); pathlib.Path('/workspace/${finish}').write_text('${result}'). Wait for that command, then reply exactly ${result}. Do not schedule it or use background processes.`;
}

test.beforeAll(async ({ request }) => {
  test.setTimeout(660_000);
  const owned = ownedBackend();
  const port = process.env.QPL_BROWSER_BACKEND_PORT ?? '8001';
  const healthUrl = `http://127.0.0.1:${port}/health`;
  const health = await (await request.get(healthUrl)).json();
  expect(health.instance_id).toMatch(/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/);
  if (!health.dispatch_capable) {
    // The production probe records its own bounded attempts; never erase its ledger.
    docker('exec', owned.name, 'python', 'scripts/check-runtime.py', '--target-instance', health.instance_id);
  }
  await expect.poll(async () => (await (await request.get(healthUrl)).json()).dispatch_capable, { timeout: 30_000 }).toBe(true);
});

test('real commands, closed-page completion, Stop, persistent files and account isolation', async ({ page, context, browser }) => {
  test.setTimeout(420_000);
  const suffix = crypto.randomUUID();
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`real-${suffix}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('real acceptance disposable password');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your assistants' })).toBeVisible();
  const models = (await (await context.request.get('/api/models')).json()).models as string[];
  expect(models.length).toBeGreaterThan(0);
  await page.getByLabel('Name', { exact: true }).fill(`Real browser ${suffix.slice(0, 8)}`);
  await page.getByLabel('Model', { exact: true }).selectOption(models.includes('gpt-6-luna') ? 'gpt-6-luna' : models[0]);
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page).toHaveURL(/\/agents\/[0-9a-f-]+$/);
  const agentUrl = page.url();
  const base = new URL(agentUrl).pathname.replace('/agents/', '/api/agents/');
  await send(page, commandPrompt('ready-start.txt', 'ready.txt', 'BROWSER_REAL_READY', 0));
  await expect.poll(() => file(context.request, base, 'ready.txt'), { timeout: 120_000 }).toBe('BROWSER_REAL_READY');
  await expect(page.locator('.reply p').filter({ hasText: /^BROWSER_REAL_READY$/ })).toBeVisible({ timeout: 60_000 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

  // Wait for an actual command-side marker before disconnecting the UI.
  await send(page, `Execute this Python code in one foreground shell command, then reply exactly BACKGROUND_READY:\nfrom pathlib import Path\nimport time\nPath('/workspace/background-start.txt').write_text('STARTED')\ndeadline = time.monotonic() + 90\nwhile not Path('/workspace/browser-release.txt').exists() and time.monotonic() < deadline:\n    time.sleep(0.2)\nassert Path('/workspace/browser-release.txt').exists(), 'Browser did not release the test'\nPath('/workspace/background.txt').write_text('BACKGROUND_READY')\nDo not create browser-release.txt yourself; the test operator will upload it after disconnecting.`);
  await expect.poll(() => file(context.request, base, 'background-start.txt'), { timeout: 120_000 }).toBe('STARTED');
  await page.close();
  expect(context.pages()).toHaveLength(0);
  expect(await file(context.request, base, 'background.txt')).toBeNull();
  const session = await (await context.request.get('/api/auth/session')).json();
  const release = await context.request.post(`${base}/files?path=browser-release.txt`, {
    headers: { Origin: new URL(agentUrl).origin, 'X-CSRF-Token': session.csrf_token, 'X-Request-ID': crypto.randomUUID(), 'Content-Type': 'application/octet-stream' },
    data: Buffer.from('RELEASE'),
  });
  expect(release.ok()).toBe(true);
  await expect.poll(() => file(context.request, base, 'background.txt'), { timeout: 90_000 }).toBe('BACKGROUND_READY');
  page = await context.newPage();
  await page.goto(agentUrl);
  await expect(page.locator('.reply p').filter({ hasText: /^BACKGROUND_READY$/ })).toBeVisible({ timeout: 60_000 });

  // A delayed side effect distinguishes stopped execution from a changed badge.
  await send(page, commandPrompt('stop-start.txt', 'must-not-exist.txt', 'STOP_FAILED', 20));
  await expect.poll(() => file(context.request, base, 'stop-start.txt'), { timeout: 120_000 }).toBe('STARTED');
  await page.getByRole('button', { name: 'Stop run', exact: true }).click();
  await expect(page.getByText('Interrupted', { exact: true })).toBeVisible({ timeout: 60_000 });
  await expect.poll(async () => (await (await context.request.get(base)).json()).paused).toBe(true);
  await page.waitForTimeout(25_000);
  expect(await file(context.request, base, 'must-not-exist.txt')).toBeNull();
  await page.reload();
  await expect(page.locator('.reply p').filter({ hasText: /^BROWSER_REAL_READY$/ })).toBeVisible();
  expect(await file(context.request, base, 'ready.txt')).toBe('BROWSER_REAL_READY');

  // A different authenticated browser must not inherit the first account's records.
  const stranger = await browser.newContext({ baseURL: new URL(agentUrl).origin });
  try {
    const other = await stranger.newPage();
    await other.goto('/register');
    await other.getByLabel('Email', { exact: true }).fill(`other-${suffix}@example.test`);
    await other.getByLabel('Password', { exact: true }).fill('different disposable password');
    await other.getByRole('button', { name: 'Create account', exact: true }).click();
    await expect(other.getByRole('heading', { name: 'Your assistants' })).toBeVisible();
    for (const path of [base, `${base}/chat`, `${base}/jobs`, `${base}/files/preview?path=ready.txt`]) expect((await stranger.request.get(path)).status()).toBe(404);
    await other.goto(agentUrl);
    await expect(other.getByRole('alert')).toContainText('Assistant not found');
  } finally { await stranger.close(); }
});
