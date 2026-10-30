import type { ReactNode } from "react";
import { Children, cloneElement, isValidElement, useId } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowUpRight,
  Check,
  AlertCircle,
  LoaderCircle,
  X,
} from "lucide-react";
import * as Dialog from "@radix-ui/react-dialog";
import { useReducedMotion } from "motion/react";
import { AnimatedGroup } from "./motion-primitives/animated-group";

export function Brand() {
  return (
    <Link className="brand" to="/" aria-label="TA-EFL Health home">
      <span className="brand-symbol">
        <Activity size={23} />
      </span>
      <span>
        TA-EFL<span className="brand-health"> Health</span>
        <small>INTELLIGENCE, TOGETHER.</small>
      </span>
    </Link>
  );
}
export function Intro({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-intro">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="muted">{description}</p>
      </div>
      {action}
    </header>
  );
}
export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={`card ${className}`}>{children}</section>;
}
export function CardHead({
  title,
  sub,
  action,
}: {
  title: string;
  sub?: string;
  action?: ReactNode;
}) {
  return (
    <div className="card-head">
      <div>
        <h2>{title}</h2>
        {sub && <p className="muted small">{sub}</p>}
      </div>
      {action}
    </div>
  );
}
export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return (
    <span className={`badge ${tone}`}>
      {tone === "success" ? (
        <Check size={12} />
      ) : tone === "error" ? (
        <AlertCircle size={12} />
      ) : null}
      {children}
    </span>
  );
}
export function Status({ value }: { value: string }) {
  return (
    <Badge
      tone={
        ["completed", "valid"].includes(value)
          ? "success"
          : ["failed", "interrupted", "invalid"].includes(value)
            ? "error"
            : value === "running"
              ? "info"
              : "neutral"
      }
    >
      {value.replaceAll("_", " ")}
    </Badge>
  );
}
export function Metric({
  title,
  value,
  note,
  icon,
}: {
  title: string;
  value: ReactNode;
  note: string;
  icon: ReactNode;
}) {
  return (
    <Card className="metric">
      <div className="metric-label">
        {title}
        <span className="metric-icon">{icon}</span>
      </div>
      <strong>{value}</strong>
      <p>{note}</p>
    </Card>
  );
}
export function Enter({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  const reduced = useReducedMotion();
  return reduced ? (
    <div className={className}>{children}</div>
  ) : (
    <AnimatedGroup
      className={className}
      variants={{
        container: {
          hidden: { opacity: 1 },
          visible: { opacity: 1, transition: { staggerChildren: 0.06 } },
        },
        item: {
          hidden: { opacity: 1, y: 8 },
          visible: { opacity: 1, y: 0, transition: { duration: 0.25 } },
        },
      }}
    >
      {children}
    </AnimatedGroup>
  );
}
export function Notice({
  children,
  error = false,
}: {
  children: ReactNode;
  error?: boolean;
}) {
  return (
    <div
      className={`notice ${error ? "error" : ""}`}
      role={error ? "alert" : "note"}
    >
      <AlertCircle size={17} />
      <div>{children}</div>
    </div>
  );
}
export function Loading() {
  return (
    <div className="loading" role="status">
      <LoaderCircle className="spin" size={22} /> Loading research data…
    </div>
  );
}
export function Empty({
  title = "No experiments yet",
  children,
}: {
  title?: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <img src="/assets/empty-experiment.svg" alt="" />
      <h3>{title}</h3>
      <p className="muted">
        {children ?? "Start a simulation to create a real result."}
      </p>
    </div>
  );
}
export function RouteError({
  error,
  retry,
}: {
  error: Error;
  retry?: () => void;
}) {
  return (
    <Card>
      <Notice error>{error.message}</Notice>
      {retry && (
        <button className="button secondary" onClick={retry}>
          Try again
        </button>
      )}
    </Card>
  );
}
export function Modal({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog-content">
          <Dialog.Title>{title}</Dialog.Title>
          <Dialog.Description className="muted">
            {description}
          </Dialog.Description>
          <Dialog.Close
            className="icon-button dialog-close"
            aria-label="Close dialog"
          >
            <X size={20} />
          </Dialog.Close>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
export function TextLink({
  to,
  children,
}: {
  to: string;
  children: ReactNode;
}) {
  return (
    <Link className="text-link" to={to}>
      {children}
      <ArrowUpRight size={15} />
    </Link>
  );
}
export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {Children.map(children, (child) =>
        isValidElement<Record<string, unknown>>(child) &&
        typeof child.type === "string" &&
        ["input", "select", "textarea"].includes(child.type)
          ? cloneElement(child, {
              id,
              "aria-describedby": hint ? `${id}-hint` : undefined,
            })
          : child,
      )}
      {hint && <small id={`${id}-hint`}>{hint}</small>}
    </div>
  );
}
