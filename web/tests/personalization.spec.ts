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
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

test("hospital calibration page shows saved study results from the backend", async ({
  page,
}) => {
  await login(page);
  const reports = await (
    await page.request.get("/api/personalization/reports")
  ).json();
  expect(reports.length).toBeGreaterThan(0);
  const report = await (
    await page.request.get(`/api/personalization/reports/${reports[0].name}`)
  ).json();
  await page.goto("/app/personalization");
  await expect(
    page.getByRole("heading", { name: "Hospital calibration", exact: true }),
  ).toBeVisible();
  const g4 = report.groups.find(
    (g: { group: string; brier: (number | null)[] }) =>
      g.group === "G4" && g.brier[0] != null,
  );
  const cell = `${g4.brier[0].toFixed(3)} ± ${g4.brier[1].toFixed(3)}`;
  await expect(page.getByRole("cell", { name: cell }).first()).toBeVisible();
  await page.getByRole("button", { name: "Details" }).first().click();
  await expect(
    page.getByRole("heading", { name: "Per-hospital results" }),
  ).toBeVisible();
  await expect(page.getByText("Global model unchanged")).toBeVisible();
  await accessible(page);
  await page.screenshot({
    path: "test-results/personalization.png",
    fullPage: true,
  });
});

test("personalised prediction compares global and hospital-calibrated probabilities", async ({
  page,
}) => {
  await login(page);
  const runs = await (
    await page.request.get("/api/personalization/experiments?dataset=heart")
  ).json();
  const run = runs.find(
    (r: { n_personalized: number; global_experiment: { strategy: string } }) =>
      r.n_personalized > 0 && r.global_experiment.strategy === "trust",
  );
  expect(run).toBeTruthy();
  const hospital = run.hospitals.find(
    (h: { status: string }) => h.status === "personalized",
  ).hospital_id;
  await page.goto("/app/prediction");
  await expect(
    page.getByText("Personalised mode needs a hospital calibration", {
      exact: false,
    }),
  ).toBeVisible();
  await expect(
    page.locator(`option[value="${run.global_experiment.id}"]`),
  ).toContainText("hospital calibration available");
  await page.getByLabel("Saved model").selectOption(run.global_experiment.id);
  await expect(
    page.getByText("Personalised mode needs a hospital calibration", {
      exact: false,
    }),
  ).toHaveCount(0);
  await page.getByLabel("Prediction mode").selectOption("personalized");
  await page.getByLabel("Calibration run").selectOption(run.id);
  await page.getByLabel("Hospital").selectOption(hospital);
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
  for (const [name, value] of Object.entries(values)) {
    const field = page.locator(`[name="${name}"]`);
    if ((await field.evaluate((e) => e.tagName)) === "SELECT")
      await field.selectOption(value);
    else await field.fill(value);
  }
  await page.getByRole("button", { name: "Predict & explain" }).click();
  await expect(
    page.getByRole("heading", {
      name: "Global vs hospital-calibrated probability",
    }),
  ).toBeVisible();
  const session = await (await page.request.get("/api/auth/session")).json();
  const expected = await (
    await page.request.post("/api/personalization/predictions", {
      headers: { "X-CSRF-Token": session.csrf },
      data: {
        experiment_id: run.global_experiment.id,
        personalization_id: run.id,
        hospital_id: hospital,
        values,
        explain: false,
      },
    })
  ).json();
  await expect(
    page.getByText(pct(expected.personalized.probability_positive), {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText("They do not explain the hospital calibration step", {
      exact: false,
    }),
  ).toBeVisible();
  await accessible(page);
  await page.screenshot({
    path: "test-results/personalized-prediction.png",
    fullPage: true,
  });
});
