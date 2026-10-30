import { Link } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  ShieldCheck,
  Network,
  ScanLine,
  ChartNoAxesCombined,
  LockKeyhole,
  Layers,
} from "lucide-react";
import { Brand, Enter } from "../components/ui";
import { ThemeToggle } from "../components/theme-control";
export default function Landing() {
  return (
    <div className="landing">
      <header className="landing-nav">
        <Brand />
        <nav aria-label="Product navigation">
          <ThemeToggle onDark />
          <a href="#approach">The approach</a>
          <a href="#workspace">The workspace</a>
          <Link className="button glass" to="/login">
            Open workspace <ArrowUpRight size={16} />
          </Link>
        </nav>
      </header>
      <main>
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">
              <span className="tiny-line" /> TRUST-AWARE FEDERATED LEARNING
            </p>
            <h1>
              Better intelligence.
              <br />
              Built <span>together.</span>
            </h1>
            <p className="hero-description">
              Explore what healthcare AI can learn together. A research
              workspace for collaborative learning, transparent predictions, and
              trust you can inspect.
            </p>
            <div className="hero-actions">
              <Link className="button mint" to="/login">
                Enter the research workspace <ArrowRight size={18} />
              </Link>
              <a className="quiet-link" href="#approach">
                Explore the approach <ArrowUpRight size={15} />
              </a>
            </div>
            <div className="hero-note">
              <ShieldCheck size={17} /> Real Python models. Traceable
              experiments.
            </div>
          </div>
          <div className="hero-art">
            <img
              src="/assets/research-orbit.svg"
              alt="Abstract orbital illustration of connected research systems"
            />
            <div className="orbit-label label-top">
              <span className="orbit-mark" />
              <div>
                Shared learning<small>Individual contributions</small>
              </div>
            </div>
            <div className="orbit-label label-bottom">
              <Layers size={20} />
              <div>
                One research workspace
                <small>Training → trust → explanation</small>
              </div>
            </div>
            <span className="art-caption">
              A conceptual view of collaborative intelligence
            </span>
          </div>
        </section>
        <div className="principles">
          <span>
            <Network size={19} /> Collaborative by design
          </span>
          <span>
            <ScanLine size={19} /> Explainable by default
          </span>
          <span>
            <ChartNoAxesCombined size={19} /> Grounded in evidence
          </span>
          <span>
            <LockKeyhole size={19} /> Local research workspace
          </span>
        </div>
        <section id="approach" className="landing-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">FROM CONTRIBUTIONS TO CONFIDENCE</p>
              <h2>See the whole learning story.</h2>
            </div>
            <p>
              Follow every experiment from its local training data to the
              signals behind its predictions.
            </p>
          </div>
          <Enter className="feature-grid">
            {[
              {
                n: "01",
                icon: <Network />,
                title: "Learn across clients",
                text: "Simulate hospital clients using Flower. Compare trust-aware aggregation with FedAvg and coordinate-wise median.",
              },
              {
                n: "02",
                icon: <ShieldCheck />,
                title: "Make trust inspectable",
                text: "Examine validation performance, update similarity, trust scores and each client’s actual contribution to the model.",
              },
              {
                n: "03",
                icon: <ScanLine />,
                title: "Explain the prediction",
                text: "Use saved preprocessing and real SHAP values to see what raised or lowered a model’s positive-class score.",
              },
            ].map((f) => (
              <article className="feature" key={f.n}>
                <div className="feature-top">
                  {f.icon}
                  <span>{f.n}</span>
                </div>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </article>
            ))}
          </Enter>
        </section>
        <section id="workspace" className="workspace-section">
          <div>
            <p className="eyebrow">A WORKSPACE FOR RIGOR</p>
            <h2>
              Every result has
              <br />a research trail.
            </h2>
            <p>
              Keep datasets, configurations, model artifacts and held-out
              evaluations connected. Search historical experiments, compare
              compatible runs and export reproducible reports.
            </p>
            <Link className="text-link" to="/login">
              Explore your experiments <ArrowUpRight size={18} />
            </Link>
          </div>
          <div className="architecture-card">
            <p className="eyebrow">THE ACTUAL ARCHITECTURE</p>
            <div className="architecture-step">
              <span>01</span>
              <div>
                <strong>Local client training</strong>
                <small>Raw rows stay inside simulated clients</small>
              </div>
            </div>
            <div className="architecture-step">
              <span>02</span>
              <div>
                <strong>Trust-aware coordination</strong>
                <small>Model updates and aggregate statistics</small>
              </div>
            </div>
            <div className="architecture-step">
              <span>03</span>
              <div>
                <strong>Held-out evaluation</strong>
                <small>Saved model · test metrics · SHAP</small>
              </div>
            </div>
            <p className="small muted">
              All clients run in one Python process. This is a simulation,
              without secure aggregation or differential privacy.
            </p>
          </div>
        </section>
        <section className="landing-section product-preview">
          <div className="section-heading">
            <div>
              <p className="eyebrow">THE RESEARCH WORKSPACE</p>
              <h2>A clearer view of every experiment.</h2>
            </div>
            <Link className="text-link" to="/login">
              Open your workspace <ArrowUpRight size={16} />
            </Link>
          </div>
          <figure>
            <div className="preview-toolbar">
              <span>TA-EFL Health / Research overview</span>
              <span>Historical snapshot</span>
            </div>
            <div className="preview-image">
              <img
                src="/assets/workspace-preview.png"
                alt="Actual TA-EFL research dashboard showing preserved experiments, validation history and a saved simulated network"
                loading="lazy"
              />
            </div>
            <figcaption>
              Captured from the working application using a copy of existing
              research records. This is a saved product preview, not live
              activity or a clinical performance claim.
            </figcaption>
          </figure>
        </section>
        <section className="landing-cta">
          <img src="/assets/layered-waves.svg" alt="" />
          <div>
            <p className="eyebrow">YOUR NEXT QUESTION STARTS HERE</p>
            <h2>
              Make the experiment.
              <br />
              Understand the result.
            </h2>
            <Link className="button mint" to="/login">
              Open workspace <ArrowRight size={18} />
            </Link>
          </div>
        </section>
      </main>
      <footer className="landing-footer">
        <Brand />
        <p>
          Research software. Model outputs are not diagnoses
          <br />
          or clinically calibrated risk estimates.
        </p>
        <span>TA-EFL Health · Research edition</span>
      </footer>
    </div>
  );
}
