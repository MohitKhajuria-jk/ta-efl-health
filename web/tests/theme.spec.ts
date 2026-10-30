import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function accessible(page: Page) {
  const { violations } = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({
        target: n.target,
        summary: n.failureSummary,
      })),
    })),
  ).toEqual([]);
}

test("theme follows the device until explicitly changed and persists across pages and tabs", async ({
  page,
  context,
}) => {
  await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
  await page.goto("/login");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(
    page.getByRole("button", { name: "Switch to light theme" }),
  ).toBeVisible();
  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await accessible(page);
  await page.screenshot({
    path: "test-results/login-dark.png",
    fullPage: true,
  });
  const other = await context.newPage();
  await other.goto("/");
  await expect(other.locator("html")).toHaveAttribute("data-theme", "dark");
  await other.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await other.close();
});

test("theme controls work when browser storage is unavailable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => {
      throw new DOMException("Storage disabled", "SecurityError");
    };
    Storage.prototype.setItem = () => {
      throw new DOMException("Storage disabled", "SecurityError");
    };
  });
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/login");
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByLabel("Username")).toBeVisible();
});

test("dark theme covers research pages and settings can restore system appearance", async ({
  page,
}) => {
  test.setTimeout(90000);
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await page.goto("/login");
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await page.getByLabel("Username").fill("browser-researcher");
  await page
    .getByLabel("Password", { exact: true })
    .fill("browser-test-only-password");
  await page
    .getByRole("button", { name: /Create account|Enter workspace/ })
    .click();
  await expect(
    page.getByRole("heading", { name: "Research overview", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".recharts-area-curve")).toBeVisible();
  await accessible(page);
  await page.screenshot({ path: "test-results/dashboard-dark.png" });
  for (const [route, heading] of [
    ["simulation", "Federated learning"],
    ["trust", "Trust monitoring"],
    ["prediction", "Risk prediction"],
    ["analytics", "Research analytics"],
    ["datasets", "Dataset library"],
    ["history", "Experiment history"],
    ["settings", "Settings & system"],
  ]) {
    await page.goto(`/app/${route}`);
    await expect(
      page.getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    await accessible(page);
  }
  await page.getByRole("radio", { name: "Light", exact: true }).check();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("radio", { name: "System", exact: true }).check();
  await page.emulateMedia({ colorScheme: "dark" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(
    page.getByRole("radio", { name: "System", exact: true }),
  ).toBeChecked();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/app/datasets");
  await expect(
    page.getByRole("heading", { name: "Dataset library" }),
  ).toBeVisible();
  await expect(page.locator("body")).toHaveJSProperty("scrollWidth", 390);
  await page.getByRole("button", { name: "Open navigation" }).click();
  await accessible(page);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.screenshot({
    path: "test-results/datasets-dark-mobile.png",
    fullPage: true,
  });
});

test("dark theme landing remains readable and its mobile switch remains accessible", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
  await page.goto("/");
  await accessible(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("body")).toHaveJSProperty("scrollWidth", 390);
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await accessible(page);
});
