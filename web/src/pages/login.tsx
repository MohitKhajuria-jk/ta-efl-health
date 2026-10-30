import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Navigate, useNavigate } from "react-router-dom";
import { ArrowRight, LockKeyhole } from "lucide-react";
import { session, post, setSession } from "../api";
import type { Session } from "../types";
import { Brand, Field, Notice, Loading, RouteError } from "../components/ui";
import { ThemeToggle } from "../components/theme-control";
const credentials = z.object({
  username: z
    .string()
    .min(3)
    .max(32)
    .regex(
      /^[A-Za-z0-9_.-]+$/,
      "Use letters, numbers, dots, dashes or underscores",
    ),
  password: z.string().min(8).max(128),
});
export default function Login() {
  const query = useQuery({ queryKey: ["session"], queryFn: session, retry: 1 });
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [error, setError] = useState("");
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<z.infer<typeof credentials>>({
    resolver: zodResolver(credentials),
  });
  if (query.data?.authenticated) return <Navigate to="/app" replace />;
  const first = query.data?.registration_open;
  async function submit(values: z.infer<typeof credentials>) {
    setError("");
    try {
      const s = await post<Session>(
        first ? "/auth/register" : "/auth/login",
        values,
      );
      setSession(s);
      qc.setQueryData(["session"], s);
      navigate("/app");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <main className="auth-page">
      <aside className="auth-story">
        <Brand />
        <div>
          <p className="eyebrow">WELCOME TO YOUR RESEARCH WORKSPACE</p>
          <h1>
            Intelligence
            <br />
            with a clearer
            <br />
            <span>perspective.</span>
          </h1>
          <p>
            Train collaboratively. Inspect every contribution.
            <br />
            Understand the signals behind each prediction.
          </p>
        </div>
        <img src="/assets/research-orbit.svg" alt="" />
        <small>Trust-Aware Explainable Federated Learning</small>
      </aside>
      <section className="auth-form">
        <div className="auth-theme-control">
          <ThemeToggle />
        </div>
        <div className="auth-form-inner">
          <div className="auth-icon">
            <LockKeyhole size={24} />
          </div>
          <p className="eyebrow">TA-EFL HEALTH</p>
          <h1>{first ? "Create your workspace" : "Welcome back."}</h1>
          <p className="muted">
            {first
              ? "Create the first local account to begin your research."
              : "Sign in with your existing workstation account."}
          </p>
          {query.isPending ? (
            <Loading />
          ) : query.error ? (
            <RouteError error={query.error} retry={() => query.refetch()} />
          ) : (
            <form onSubmit={handleSubmit(submit)}>
              <Field label="Username">
                <input
                  autoComplete="username"
                  {...register("username")}
                  aria-invalid={!!errors.username}
                />
                {errors.username && (
                  <span className="field-error">{errors.username.message}</span>
                )}
              </Field>
              <Field label="Password">
                <input
                  type="password"
                  autoComplete={first ? "new-password" : "current-password"}
                  {...register("password")}
                  aria-invalid={!!errors.password}
                />
                {errors.password && (
                  <span className="field-error">{errors.password.message}</span>
                )}
              </Field>
              {error && <Notice error>{error}</Notice>}
              <button className="button primary full" disabled={isSubmitting}>
                {isSubmitting
                  ? "Verifying…"
                  : first
                    ? "Create account"
                    : "Enter workspace"}
                <ArrowRight size={17} />
              </button>
            </form>
          )}
          <p className="auth-footnote">
            <LockKeyhole size={14} /> Local account · protected session
          </p>
          <p className="small muted">
            Research use only. Predictions are model outputs, not medical advice
            or diagnoses.
          </p>
        </div>
      </section>
    </main>
  );
}
