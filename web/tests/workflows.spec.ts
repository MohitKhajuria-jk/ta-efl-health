import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
const account = {
  username: "browser-researcher",
  password: "browser-test-only-password",
};
async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Username").fill(account.username);
  await page.getByLabel("Password", { exact: true }).fill(account.password);
  await page
    .getByRole("button", { name: /Enter workspace|Create account/ })
    .click();
  await expect(
    page.getByRole("heading", { name: "Research overview", exact: true }),
  ).toBeVisible();
}
async function accessible(page: Page) {
  const result = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    result.violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
}

test("landing has original assets, responsive layout and accessible calls to action", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Better intelligence. Built together." }),
  ).toBeVisible();
  await expect(
    page.locator('img[src="/assets/research-orbit.svg"]'),
  ).toBeVisible();
  await accessible(page);
  await page.screenshot({
    path: "test-results/landing-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("body")).toHaveJSProperty("scrollWidth", 390);
  await page.screenshot({
    path: "test-results/landing-mobile.png",
    fullPage: true,
  });
});

test("existing authentication and every workspace route load without browser exceptions", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await login(page);
  const dashboard = await (await page.request.get("/api/dashboard")).json();
  await expect(
    page
      .getByText(dashboard.total_experiments.toLocaleString(), { exact: true })
      .first(),
  ).toBeVisible();
  await accessible(page);
  await page.screenshot({
    path: "test-results/dashboard-desktop.png",
    fullPage: true,
  });
  await accessible(page);
  for (const [path, title] of [
    ["simulation", "Federated learning"],
    ["network", "Hospital network"],
    ["trust", "Trust monitoring"],
    ["prediction", "Risk prediction"],
    ["analytics", "Research analytics"],
    ["datasets", "Dataset library"],
    ["history", "Experiment history"],
    ["settings", "Settings & system"],
  ]) {
    await page.goto(`/app/${path}`);
    await expect(
      page.getByRole("heading", { name: title, exact: true }),
    ).toBeVisible();
    await accessible(page);
  }
  expect(errors).toEqual([]);
});

test("a real simulation runs through the web and displays normalized trust weights", async ({
  page,
}) => {
  await login(page);
  await page.goto("/app/simulation");
  await page
    .getByLabel("Dataset", { exact: true })
    .selectOption("breast_cancer");
  await page.getByLabel("Simulated clients").fill("4");
  await page.getByLabel("Federated rounds").fill("6");
  await page.getByLabel("Local epochs", { exact: true }).fill("2");
  await page.getByLabel("Reproducibility seed").fill("987");
  await page
    .getByLabel("Experiment label", { exact: false })
    .fill("Browser verification · genuine engine");
  await page
    .getByRole("button", { name: "Start simulation", exact: true })
    .click();
  await expect(
    page.getByRole("link", { name: "Inspect saved result" }),
  ).toBeVisible({ timeout: 30000 });
  await expect(
    page.getByText("Weight sum: 1.000000.", { exact: false }),
  ).toBeVisible();
  const jobs = await (await page.request.get("/api/federation")).json();
  expect(jobs[0].status).toBe("completed");
  expect(jobs[0].completed_rounds).toBe(6);
  const detail = await (
    await page.request.get(`/api/federation/${jobs[0].id}`)
  ).json();
  expect(
    detail.events.some((e: { type: string }) => e.type === "client_training"),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/simulation.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Inspect saved result" }).click();
  await expect(
    page.getByRole("heading", {
      name: "Browser verification · genuine engine",
    }),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "PDF report" }).click();
  expect((await download).suggestedFilename()).toMatch(/\.pdf$/);
});

test("prediction uses a saved model and produces real SHAP contributions", async ({
  page,
}) => {
  await login(page);
  await page.goto("/app/prediction");
  await expect(
    page.getByRole("heading", { name: "Research input profile" }),
  ).toBeVisible();
  const datasets = await (await page.request.get("/api/datasets")).json();
  const spec = datasets.find((d: { key: string }) => d.key === "heart");
  const values: Record<string, string> = {
    Age: "54",
    Sex: "M",
    ChestPainType: "ATA",
    RestingBP: "120",
    Cholesterol: "200",
    FastingBS: "0",
    RestingECG: "Normal",
    MaxHR: "150",
    ExerciseAngina: "N",
    Oldpeak: "1",
    ST_Slope: "Flat",
  };
  for (const feature of spec.features) {
    const field = page.locator(`[name="${feature.name}"]`);
    if (feature.kind === "categorical" || feature.kind === "binary")
      await field.selectOption(values[feature.name]);
    else await field.fill(values[feature.name]);
  }
  await page.getByRole("button", { name: "Predict & explain" }).click();
  await expect(
    page.getByRole("heading", { name: "Why this prediction?" }),
  ).toBeVisible();
  await expect(
    page.getByText("Additivity is checked in Python.", { exact: false }),
  ).toBeVisible();
  await accessible(page);
  await page.screenshot({
    path: "test-results/prediction.png",
    fullPage: true,
  });
});

test("history search, pagination and compatible experiment comparison work", async ({
  page,
}) => {
  await login(page);
  await page.goto("/app/history");
  await page.getByRole("button", { name: "Next page" }).click();
  await expect(page.getByText("Page 2 of", { exact: false })).toBeVisible();
  await page
    .getByLabel("Dataset", { exact: true })
    .selectOption("breast_cancer");
  const rows = await (
    await page.request.get(
      "/api/experiments?dataset=breast_cancer&page_size=15",
    )
  ).json();
  const first = rows.items.find(
    (r: { config: { seed: number } }) => r.config.seed === 1,
  );
  const second = rows.items.find(
    (r: { id: string; config: { seed: number } }) =>
      r.config.seed === 1 && r.id !== first.id,
  );
  await page
    .getByRole("checkbox", { name: `Compare ${first.id}`, exact: true })
    .check();
  await page
    .getByRole("checkbox", { name: `Compare ${second.id}`, exact: true })
    .check();
  await page.getByRole("button", { name: "Compare results" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByText("Saved held-out test results on the same split.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByLabel("Search run ID or label").fill(first.id);
  await expect(page.getByText("1 experiments · Page 1 of 1")).toBeVisible();
  await accessible(page);
});

test("mobile navigation, focus management and reduced motion remain usable", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await login(page);
  await expect(page.locator("body")).toHaveJSProperty("scrollWidth", 390);
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page
    .getByRole("dialog")
    .getByRole("link", { name: "Datasets", exact: true })
    .click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Dataset library" }),
  ).toBeVisible();
  await expect(page.locator("body")).toHaveJSProperty("scrollWidth", 390);
  expect(
    await page.evaluate(
      () => matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
  ).toBe(true);
  await accessible(page);
  await page.screenshot({
    path: "test-results/datasets-mobile.png",
    fullPage: true,
  });
});

test("session logout prevents access to protected workflows", async ({
  page,
}) => {
  await login(page);
  await page.getByRole("button", { name: /Researcher · Sign out/ }).click();
  await expect(
    page.getByRole("heading", { name: "Welcome back." }),
  ).toBeVisible();
  await page.goto("/app/prediction");
  await expect(
    page.getByRole("heading", { name: "Welcome back." }),
  ).toBeVisible();
  expect((await page.request.get("/api/models")).status()).toBe(401);
});

test("a real running simulation can be cancelled at a safe boundary", async ({
  page,
}) => {
  await login(page);
  await page.goto("/app/simulation");
  await page.getByLabel("Federated rounds", { exact: true }).fill("100");
  await page.getByLabel("Local epochs", { exact: true }).fill("10");
  await page
    .getByRole("button", { name: "Start simulation", exact: true })
    .click();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Cancelled", exact: true }),
  ).toBeVisible({ timeout: 30000 });
  const jobs = await (await page.request.get("/api/federation")).json();
  expect(jobs[0].status).toBe("cancelled");
  expect(jobs[0].cancel_requested).toBe(true);
  expect(jobs[0].completed_rounds).toBeLessThan(100);
});
