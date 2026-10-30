import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useState,
  type ReactNode,
} from "react";

export type ThemePreference = "light" | "dark" | "system";
const STORAGE_KEY = "ta-efl-theme";
function readPreference(): ThemePreference {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (value === "light" || value === "dark") return value;
  } catch {
    /* Appearance also works without browser storage. */
  }
  return "system";
}
const ThemeContext = createContext<{
  preference: ThemePreference;
  resolvedTheme: "light" | "dark";
  setPreference: (value: ThemePreference) => void;
} | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, updatePreference] = useState(readPreference);
  const [systemDark, setSystemDark] = useState(
    () => matchMedia("(prefers-color-scheme: dark)").matches,
  );
  const resolvedTheme =
    preference === "system" ? (systemDark ? "dark" : "light") : preference;

  useEffect(() => {
    const media = matchMedia("(prefers-color-scheme: dark)");
    const update = () => setSystemDark(media.matches);
    const syncStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY || event.key === null)
        updatePreference(readPreference());
    };
    update();
    media.addEventListener("change", update);
    window.addEventListener("storage", syncStorage);
    return () => {
      media.removeEventListener("change", update);
      window.removeEventListener("storage", syncStorage);
    };
  }, []);

  useLayoutEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = resolvedTheme;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute(
        "content",
        getComputedStyle(root).getPropertyValue("--background").trim(),
      );
  }, [resolvedTheme]);

  function setPreference(value: ThemePreference) {
    updatePreference(value);
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch {
      /* Keep the current session usable. */
    }
  }

  return (
    <ThemeContext.Provider value={{ preference, resolvedTheme, setPreference }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const value = useContext(ThemeContext);
  if (!value) throw new Error("Theme controls require ThemeProvider");
  return value;
}
