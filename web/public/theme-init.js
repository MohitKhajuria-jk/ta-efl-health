// Apply the saved appearance before React/CSS paint. Kept external for the CSP.
(() => {
  let preference = "system";
  try {
    const saved = localStorage.getItem("ta-efl-theme");
    if (["light", "dark", "system"].includes(saved)) preference = saved;
  } catch {
    // Storage can be disabled; the system preference still works.
  }
  document.documentElement.dataset.theme =
    preference === "system"
      ? matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : preference;
})();
