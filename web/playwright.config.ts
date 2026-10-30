import { defineConfig } from "@playwright/test";
import path from "node:path";
const python = path.resolve("..", ".venv", "Scripts", "python.exe");
const server = path.resolve("..", "scripts", "serve_web_qa.py");
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 60000,
  expect: { timeout: 15000 },
  reporter: [["list"], ["json", { outputFile: "test-results/results.json" }]],
  use: {
    baseURL: "http://127.0.0.1:8781",
    viewport: { width: 1440, height: 1000 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: { executablePath: process.env.QA_BROWSER },
  },
  webServer: {
    command: `"${python}" "${server}"`,
    url: "http://127.0.0.1:8781/api/health",
    timeout: 120000,
    reuseExistingServer: false,
    stdout: "ignore",
    stderr: "pipe",
  },
});
