import { expect, test } from '@playwright/test';

test('account, persisted chat, Stop, New chat, and cross-account ownership', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`ana-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your assistants' })).toBeVisible();
  await page.getByLabel('Name', { exact: true }).fill('Private study assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Private study assistant' })).toBeVisible();
  const agentUrl = page.url();
  await page.getByLabel('Message your assistant').fill('Summarize notes');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText('Test reply: Summarize notes', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.reload();
  await expect(page.getByText('Test reply: Summarize notes', { exact: true })).toBeVisible();
  await page.getByLabel('Message your assistant').fill('Wait for stop');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByRole('button', { name: 'Stop run', exact: true }).click();
  await expect(page.getByText('Interrupted', { exact: true })).toBeVisible();
  await page.getByLabel('Message your assistant').fill('Queued old context');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Queued old context', exact: true })).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'New chat', exact: true }).click();
  await expect(page.getByText('Cancelled', { exact: true })).toBeVisible();
  await expect(page.getByText('Conversation 1 · Archived', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Continue assistant' })).toBeEnabled();
  const chat = await page.request.get(`${new URL(agentUrl).pathname.replace('/agents/', '/api/agents/')}/chat`);
  expect((await chat.json()).map((run: { prompt: string }) => run.prompt)).toEqual(['Queued old context', 'Wait for stop', 'Summarize notes']);
  await page.getByRole('button', { name: 'Sign out', exact: true }).click();
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`ben-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your assistants' })).toBeVisible();
  await page.goto(agentUrl);
  await expect(page.getByRole('alert')).toContainText('Assistant not found');
  await expect(page.getByText('Summarize notes', { exact: true })).toHaveCount(0);
});

test('a lost creation response reuses its request ID and creates only one assistant', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`retry-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your assistants' })).toBeVisible();
  const requestIds: string[] = [];
  let acceptedId = '';
  await page.route('**/api/agents', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    requestIds.push(route.request().headers()['x-request-id']);
    if (requestIds.length === 1) {
      const accepted = await route.fetch();
      acceptedId = (await accepted.json()).id;
      return route.abort('failed');
    }
    return route.continue();
  });
  await page.getByLabel('Name', { exact: true }).fill('Created once');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('alert')).toBeVisible();
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/agents/${acceptedId}$`));
  expect(requestIds).toHaveLength(2);
  expect(requestIds[0]).toBe(requestIds[1]);
  const result = await page.request.get('/api/agents');
  expect((await result.json()).map((agent: { id: string }) => agent.id)).toEqual([acceptedId]);
});

test('no verified models gives an honest empty state and disables creation', async ({ page }) => {
  await page.route('**/api/models', route => route.fulfill({ json: { models: [] } }));
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`empty-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page.getByText('No models have passed the current runtime check.', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Create assistant', exact: true })).toBeDisabled();
});

test('many paused queued prompts do not block Continue behind event streams', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`queue-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Queue assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Queue assistant' })).toBeVisible();
  const agentId = page.url().split('/').pop();
  const session = await (await page.request.get('/api/auth/session')).json();
  const headers = { Origin: new URL(page.url()).origin, 'X-CSRF-Token': session.csrf_token, 'X-Request-ID': crypto.randomUUID() };
  const sent = await page.request.post(`/api/agents/${agentId}/messages`, { headers, data: { prompt: 'Wait for stop' } });
  const runId = (await sent.json()).run_id;
  await expect.poll(async () => (await (await page.request.get(`/api/runs/${runId}`)).json()).status).toBe('running');
  expect((await page.request.post(`/api/runs/${runId}/stop`, { headers })).ok()).toBe(true);
  for (let index = 0; index < 8; index++) {
    const response = await page.request.post(`/api/agents/${agentId}/messages`, {
      headers: { ...headers, 'X-Request-ID': crypto.randomUUID() }, data: { prompt: `Queued task ${index}` },
    });
    expect(response.ok()).toBe(true);
  }
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Queued task 7', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Continue assistant' }).click();
  await expect(page.getByText('Test reply: Queued task 7', { exact: true })).toBeVisible({ timeout: 5000 });
});

test('workspace upload replay, preview, exact download, confirmed replacement and deletion', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`files-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('File assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  const workspace = page.getByRole('region', { name: 'Workspace files' });
  await expect(workspace).toBeVisible();
  const data = Buffer.from('Unicode café notes\n');
  await workspace.getByLabel('Upload file (up to 10 MiB)').setInputFiles({ name: 'café.txt', mimeType: 'text/plain', buffer: data });
  const ids: string[] = [];
  await page.route('**/api/agents/*/files?*', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    ids.push(route.request().headers()['x-request-id']);
    if (ids.length === 1) { await route.fetch(); return route.abort('failed'); }
    return route.continue();
  });
  await workspace.getByRole('button', { name: 'Upload file', exact: true }).click();
  await expect(workspace.getByRole('button', { name: 'Retry file request' })).toBeVisible();
  await workspace.getByRole('button', { name: 'Retry file request' }).click();
  await expect(workspace.getByText('File uploaded.', { exact: true })).toBeVisible();
  expect(ids).toHaveLength(2); expect(ids[0]).toBe(ids[1]);
  await workspace.getByRole('button', { name: 'café.txt', exact: true }).click();
  await expect(workspace.locator('pre')).toHaveText(data.toString().trim());
  const download = await page.request.get(await workspace.getByRole('link', { name: 'Download file' }).getAttribute('href') as string);
  expect(await download.body()).toEqual(data);
  await workspace.getByLabel('Upload file (up to 10 MiB)').setInputFiles({ name: 'café.txt', mimeType: 'text/plain', buffer: Buffer.from('replacement') });
  page.once('dialog', async dialog => { expect(dialog.message()).toContain('Current SHA-256:'); await dialog.accept(); });
  await workspace.getByRole('button', { name: 'Upload file', exact: true }).click();
  await expect(workspace.getByRole('button', { name: 'café.txt', exact: true })).toBeVisible();
  await workspace.getByRole('button', { name: 'café.txt', exact: true }).click();
  await expect(workspace.locator('pre')).toHaveText('replacement');
  page.once('dialog', dialog => dialog.accept());
  await workspace.getByRole('button', { name: 'Delete selected file' }).click();
  await expect(workspace.getByText('File deleted.', { exact: true })).toBeVisible();
  await expect(workspace.getByRole('button', { name: 'café.txt', exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('scheduled tasks survive reload, replay creation, run independently and retain cancelled history', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`schedule-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Schedule assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Scheduled tasks' });
  await expect(panel).toBeVisible();
  const base = `/api/agents/${page.url().split('/').pop()}`;
  const ids: string[] = [];
  await page.route('**/api/agents/*/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    ids.push(route.request().headers()['x-request-id']);
    if (ids.length === 1) { await route.fetch(); return route.abort('failed'); }
    return route.continue();
  });
  await panel.getByLabel('Task name', { exact: true }).fill('Recurring review');
  await panel.getByLabel('Task instructions', { exact: true }).fill('Review notes');
  await panel.getByLabel('First run date and time').fill(new Date(Date.now() + 120_000).toISOString().slice(0, 19));
  await panel.getByLabel('Repeat interval in seconds (optional)').fill('60');
  await panel.getByRole('button', { name: 'Save scheduled task' }).click();
  await expect(panel.getByRole('button', { name: 'Retry saving task' })).toBeVisible();
  await panel.getByRole('button', { name: 'Retry saving task' }).click();
  await expect(panel.getByRole('heading', { name: 'Recurring review', exact: true })).toBeVisible();
  expect(ids).toHaveLength(2); expect(ids[0]).toBe(ids[1]);
  page.once('dialog', dialog => dialog.accept());
  await panel.getByRole('button', { name: 'Cancel task: Recurring review', exact: true }).click();
  await expect(panel.getByText('Created by user · Cancelled', { exact: true })).toBeVisible();
  await panel.getByLabel('Task name', { exact: true }).fill('One-time review');
  await panel.getByLabel('Task instructions', { exact: true }).fill('Summarize saved notes');
  await panel.getByLabel('Repeat interval in seconds (optional)').fill('');
  await panel.getByLabel('First run date and time').fill(new Date(Date.now() + 4_000).toISOString().slice(0, 19));
  await panel.getByRole('button', { name: 'Save scheduled task' }).click();
  await expect(panel.getByRole('heading', { name: 'One-time review', exact: true })).toBeVisible();
  await page.reload();
  await panel.getByRole('button', { name: 'View task runs: One-time review', exact: true }).click();
  await expect(panel.getByText('Test reply: Summarize saved notes', { exact: true })).toBeVisible({ timeout: 12000 });
  expect(await (await page.request.get(`${base}/chat`)).json()).toEqual([]);
  const jobs = (await (await page.request.get(`${base}/jobs`)).json()).items;
  expect(jobs).toHaveLength(2);
  const oneTime = jobs.find((job: { name: string }) => job.name === 'One-time review');
  await expect(page.getByLabel('Export: Summarize saved notes')).toBeVisible();
  await page.getByRole('button', { name: 'Sign out', exact: true }).click();
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`schedule-other-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your assistants' })).toBeVisible();
  expect((await page.request.get(`${base}/jobs`)).status()).toBe(404);
  expect((await page.request.get(`/api/jobs/${oneTime.job_id}/runs`)).status()).toBe(404);
});

test('usage limits show the retry time without claiming the assistant is paused', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`usage-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Usage assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Usage assistant', exact: true })).toBeVisible();
  await page.route(/\/api\/agents\/[^/]+$/, async route => {
    if (route.request().method() !== 'GET') return route.continue();
    const response = await route.fetch();
    const body = await response.json();
    return route.fulfill({ response, json: { ...body, run_admission: {
      limited: true, retry_at: new Date(Date.now() + 60_000).toISOString(),
      minute_count: 5, minute_limit: 5, day_count: 6, day_limit: 20,
    } } });
  });
  await page.reload();
  await expect(page.getByText(/Model run limit reached\. Queued work will be reconsidered after/)).toBeVisible();
  await expect(page.getByText('Model starts: 5/5 in the last minute · 6/20 in the last 24 hours.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Continue assistant', exact: true })).toBeDisabled();
});

test('storage protection explains blocked writes while keeping saved reads available', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`storage-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Storage assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Storage assistant', exact: true })).toBeVisible();
  await page.route(/\/api\/agents\/[^/]+$/, async route => {
    if (route.request().method() !== 'GET') return route.continue();
    const response = await route.fetch(); const body = await response.json();
    return route.fulfill({ response, json: { ...body, storage_protection: {
      active: true, reason: 'platform_quota_high', measurement_verified: true,
    } } });
  });
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('Platform storage protection');
  await expect(page.getByRole('alert')).toContainText('Saved data remains readable');
  await expect(page.getByRole('region', { name: 'Workspace files' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Scheduled tasks' })).toBeVisible();
});

test('owned lifecycle controls and response-loss context reset preserve files, schedules and pause', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`lifecycle-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Lifecycle assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Lifecycle assistant', exact: true })).toBeVisible();
  const agentId = page.url().split('/').pop(); const base = `/api/agents/${agentId}`;
  const workspace = page.getByRole('region', { name: 'Workspace files' });
  await workspace.getByLabel('Upload file (up to 10 MiB)').setInputFiles({ name: 'keep.txt', mimeType: 'text/plain', buffer: Buffer.from('keep these bytes') });
  await workspace.getByRole('button', { name: 'Upload file', exact: true }).click();
  await expect(workspace.getByText('File uploaded.', { exact: true })).toBeVisible();
  const jobs = page.getByRole('region', { name: 'Scheduled tasks' });
  await jobs.getByLabel('Task name', { exact: true }).fill('Keep schedule');
  await jobs.getByLabel('Task instructions', { exact: true }).fill('Review saved files');
  await jobs.getByLabel('First run date and time').fill(new Date(Date.now() + 120000).toISOString().slice(0, 19));
  await jobs.getByRole('button', { name: 'Save scheduled task' }).click();
  await expect(jobs.getByRole('heading', { name: 'Keep schedule', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Wake assistant', exact: true }).click();
  await expect(page.getByText('Container: running', { exact: true })).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Sleep assistant', exact: true }).click();
  await expect(page.getByText('Container: stopped', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Continue assistant', exact: true })).toBeEnabled();
  await page.getByLabel('Message your assistant').fill('Queued before reset');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Queued before reset', exact: true })).toBeVisible();
  const ids: string[] = [];
  await page.route('**/api/agents/*/reset-model-context', async route => {
    ids.push(route.request().headers()['x-request-id']);
    if (ids.length === 1) { await route.fetch(); return route.abort('failed'); }
    return route.continue();
  });
  page.once('dialog', dialog => { expect(dialog.message()).toContain('credentials'); return dialog.accept(); });
  await page.getByRole('button', { name: 'Reset model context', exact: true }).click();
  await expect(page.getByText('Unconfirmed control requests retain their original IDs. Retry the same action to inspect its result.', { exact: true })).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Reset model context', exact: true }).click();
  await expect(page.getByText('Unconfirmed control requests retain their original IDs. Retry the same action to inspect its result.', { exact: true })).toHaveCount(0);
  expect(ids).toHaveLength(2); expect(ids[0]).toBe(ids[1]);
  const details = await (await page.request.get(base)).json();
  expect(details.chat_epoch).toBe(1); expect(details.paused).toBe(true);
  expect((await (await page.request.get(`${base}/chat`)).json())[0].status).toBe('cancelled');
  expect(await (await page.request.get(`${base}/files/download?path=keep.txt`)).body()).toEqual(Buffer.from('keep these bytes'));
  expect((await (await page.request.get(`${base}/jobs`)).json()).items[0].name).toBe('Keep schedule');
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Restart assistant', exact: true }).click();
  await expect(page.getByText('Container: running', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Continue assistant', exact: true })).toBeEnabled();
  // Release this case's idle container before later instance-wide maintenance.
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Sleep assistant', exact: true }).click();
  await expect(page.getByText('Container: stopped', { exact: true })).toBeVisible();
});

test('terminal chat history exports exact JSON with a verified digest', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`archive-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Archive assistant');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await page.getByLabel('Message your assistant').fill('Keep this saved output');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText('Test reply: Keep this saved output', { exact: true })).toBeVisible();
  await page.getByLabel('Export: Keep this saved output').check();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download selected history', exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('run-history.json');
  const stream = await download.createReadStream();
  const parts: Buffer[] = [];
  for await (const chunk of stream) parts.push(Buffer.from(chunk));
  const bytes = Buffer.concat(parts);
  const data = JSON.parse(bytes.toString('utf8'));
  expect(data.format).toBe('qpl-run-archive-v1');
  expect(data.runs[0].run.prompt).toBe('Keep this saved output');
  expect(data.runs[0].events.some((event: { payload: unknown }) => JSON.stringify(event.payload).includes('Test reply: Keep this saved output'))).toBe(true);
  await expect(page.getByText(/Export verified/)).toBeVisible();
  await page.getByRole('button', { name: 'Check last downloaded archive', exact: true }).click();
  await expect(page.getByText('Cleanup check · 1 run · deletion blocked', { exact: true })).toBeVisible();
  await expect(page.getByText('Pause required', { exact: true })).toBeVisible();

  await expect(page.getByText('Test reply: Keep this saved output', { exact: true })).toBeVisible();
});


test('history selection spans pages and exports both selected roots', async ({ page }) => {
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`pages-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('History pages');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'History pages', exact: true })).toBeVisible();
  const base = `/api/agents/${page.url().split('/').pop()}`;
  const session = await (await page.request.get('/api/auth/session')).json();
  for (let index = 0; index < 22; index++) {
    const response = await page.request.post(`${base}/messages`, {
      headers: { Origin: new URL(page.url()).origin, 'X-CSRF-Token': session.csrf_token, 'X-Request-ID': crypto.randomUUID() },
      data: { prompt: `Archive page ${index}` },
    });
    expect(response.ok()).toBe(true);
  }
  await expect(page.getByLabel('Export: Archive page 21', { exact: true })).toBeVisible({ timeout: 15000 });
  await page.getByLabel('Export: Archive page 21', { exact: true }).check();
  const failedPage = '**/api/agents/*/runs?archive_roots=true&limit=20&cursor=*';
  await page.route(failedPage, route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ code: 'test_unavailable', message: 'History temporarily unavailable.' }) }));
  await page.getByRole('button', { name: 'Next history page', exact: true }).click();
  await expect(page.getByText('History temporarily unavailable.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Next history page', exact: true })).toBeDisabled();
  await expect(page.getByLabel('Export: Archive page 21', { exact: true })).toHaveCount(0);
  await page.unroute(failedPage);

  await page.getByLabel('Export: Archive page 0', { exact: true }).check();
  await expect(page.getByText('2/100 selected across pages · Page 2', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Previous history page', exact: true }).click();
  await expect(page.getByLabel('Export: Archive page 21', { exact: true })).toBeChecked();
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/runs/archive/export'));
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download selected history', exact: true }).click();
  const response = await responsePromise; await downloadPromise;
  const archive = await response.json();
  expect(archive.runs.map((item: { run: { prompt: string } }) => item.run.prompt).sort()).toEqual(['Archive page 0', 'Archive page 21']);
});

test('confirmed archive purge replays its lost response after reload and keeps workspace', async ({ page }) => {
  await page.goto('/register');
  const email = `purge-${crypto.randomUUID()}@example.test`;
  await page.getByLabel('Email', { exact: true }).fill(email);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Confirmed cleanup');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Confirmed cleanup', exact: true })).toBeVisible();
  const assistantUrl = page.url();
  await page.getByLabel('Message your assistant').fill('Remove this saved record');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText('Test reply: Remove this saved record', { exact: true })).toBeVisible();
  await page.getByLabel('Upload file (up to 10 MiB)').setInputFiles({ name: 'kept.txt', mimeType: 'text/plain', buffer: Buffer.from('keep exact workspace bytes') });
  await page.getByRole('button', { name: 'Upload file', exact: true }).click();
  await expect(page.getByText('kept.txt', { exact: true })).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Sleep assistant', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Continue assistant', exact: true })).toBeEnabled();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'New chat', exact: true }).click();
  await page.getByLabel('Export: Remove this saved record', { exact: true }).check();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download selected history', exact: true }).click(); await download;
  await page.getByRole('button', { name: 'Check last downloaded archive', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Permanently delete downloaded records', exact: true })).toBeEnabled();
  const ids: string[] = [];
  await page.route('**/api/runs/archive/purge', async route => {
    ids.push(route.request().headers()['x-request-id']);
    if (ids.length === 1) { await route.fetch(); return route.abort('failed'); }
    return route.continue();
  });
  page.once('dialog', dialog => { expect(dialog.message()).toContain('SHA-256'); return dialog.accept(); });
  await page.getByRole('button', { name: 'Permanently delete downloaded records', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Retry saved cleanup request', exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Retry saved cleanup request', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Sign out', exact: true }).click();
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill(`purge-other-${crypto.randomUUID()}@example.test`);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill('Other account');
  await page.getByRole('button', { name: 'Create assistant', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Other account', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Retry saved cleanup request', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Sign out', exact: true }).click();
  await page.goto('/login');
  await page.getByLabel('Email', { exact: true }).fill(email);
  await page.getByLabel('Password', { exact: true }).fill('correct horse battery');
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your assistants', exact: true })).toBeVisible();
  await page.goto(assistantUrl);
  await page.getByRole('button', { name: 'Retry saved cleanup request', exact: true }).click();
  await expect(page.getByText('Saved records deleted. Space recovery complete.', { exact: true })).toBeVisible();
  expect(ids).toHaveLength(2); expect(ids[0]).toBe(ids[1]);
  await expect(page.getByLabel('Export: Remove this saved record', { exact: true })).toHaveCount(0);
  await expect(page.getByText('kept.txt', { exact: true })).toBeVisible();
});
