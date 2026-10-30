import { Moon, Sun, Monitor } from "lucide-react";
import { useTheme } from "../theme";

export function ThemeToggle({ onDark = false }: { onDark?: boolean }) {
  const { resolvedTheme, setPreference } = useTheme();
  const next = resolvedTheme === "dark" ? "light" : "dark";
  const Icon = next === "dark" ? Moon : Sun;
  return (
    <button
      type="button"
      className={`theme-toggle ${onDark ? "theme-toggle-on-dark" : ""}`}
      onClick={() => setPreference(next)}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      <Icon size={17} aria-hidden="true" />
      <span>{next === "dark" ? "Dark mode" : "Light mode"}</span>
    </button>
  );
}

export function ThemeOptions() {
  const { preference, setPreference } = useTheme();
  return (
    <fieldset className="theme-options">
      <legend className="sr-only">Color theme</legend>
      {(
        [
          ["light", "Light", Sun],
          ["dark", "Dark", Moon],
          ["system", "System", Monitor],
        ] as const
      ).map(([value, label, Icon]) => (
        <label
          key={value}
          className={`theme-option ${preference === value ? "selected" : ""}`}
        >
          <input
            className="theme-option-input"
            type="radio"
            name="color-theme"
            value={value}
            checked={preference === value}
            onChange={() => setPreference(value)}
          />
          <Icon size={20} aria-hidden="true" />
          <span>{label}</span>
        </label>
      ))}
    </fieldset>
  );
}
