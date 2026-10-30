import { test, expect } from "@playwright/test";

test("login explains an incompatible backend and recovers with retry", async ({
  page,
}) => {
  await page.route("**/api/auth/session", (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "not found" }),
    }),
  );
  await page.goto("/login");
  await expect(
    page.getByText("The sign-in service is unavailable.", { exact: false }),
  ).toBeVisible();
  await expect(page.getByLabel("Username")).not.toBeVisible();
  await page.unroute("**/api/auth/session");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByLabel("Username")).toBeVisible();
  await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Create account|Enter workspace/ }),
  ).toBeVisible();
});

test("login can recover after the backend connection fails", async ({
  page,
}) => {
  await page.route("**/api/auth/session", (route) =>
    route.abort("connectionrefused"),
  );
  await page.goto("/login");
  await expect(
    page.getByText("The research API is unavailable.", { exact: false }),
  ).toBeVisible();
  await page.unroute("**/api/auth/session");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByLabel("Username")).toBeVisible();
});
