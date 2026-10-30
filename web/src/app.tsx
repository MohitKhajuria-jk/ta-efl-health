import { useState, Suspense, lazy } from "react";
import {
  Link,
  NavLink,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  LayoutDashboard,
  Network,
  FlaskConical,
  ShieldCheck,
  ScanLine,
  ChartNoAxesCombined,
  Database,
  History,
  Settings,
  LogOut,
  ArrowUpRight,
  Menu,
  Search,
  ChevronRight,
  Activity,
  SlidersHorizontal,
} from "lucide-react";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import { session, post, useData } from "./api";
import { Brand, Loading, Modal, RouteError } from "./components/ui";
import Landing from "./pages/landing";
import Login from "./pages/login";
import { ThemeToggle } from "./components/theme-control";
const Dashboard = lazy(() => import("./pages/dashboard"));
const Simulation = lazy(() => import("./pages/simulation"));
const Trust = lazy(() => import("./pages/trust"));
const Prediction = lazy(() => import("./pages/prediction"));
const HistoryPage = lazy(() => import("./pages/history"));
const Analytics = lazy(() => import("./pages/analytics"));
const Datasets = lazy(() => import("./pages/datasets"));
const SettingsPage = lazy(() => import("./pages/settings"));
const ExperimentDetail = lazy(() => import("./pages/experiment"));
const Personalization = lazy(() => import("./pages/personalization"));
const items = [
  { to: "/app", label: "Overview", icon: LayoutDashboard },
  { to: "/app/simulation", label: "Federated learning", icon: FlaskConical },
  { to: "/app/network", label: "Hospital network", icon: Network },
  { to: "/app/trust", label: "Trust monitoring", icon: ShieldCheck },
  { to: "/app/prediction", label: "Risk prediction", icon: ScanLine },
  {
    to: "/app/personalization",
    label: "Hospital calibration",
    icon: SlidersHorizontal,
  },
  {
    to: "/app/analytics",
    label: "Research analytics",
    icon: ChartNoAxesCombined,
  },
  { to: "/app/datasets", label: "Datasets", icon: Database },
  { to: "/app/history", label: "Experiment history", icon: History },
  { to: "/app/settings", label: "Settings", icon: Settings },
];
function Shell() {
  const auth = useQuery({
    queryKey: ["session"],
    queryFn: session,
    retry: 1,
    refetchInterval: 60000,
  });
  const health = useData<{ status: string }>("/health", 15000);
  const location = useLocation();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [menu, setMenu] = useState(false);
  const reduced = useReducedMotion();
  if (auth.isPending) return <Loading />;
  if (auth.error)
    return <RouteError error={auth.error} retry={() => auth.refetch()} />;
  if (!auth.data?.authenticated) return <Navigate to="/login" replace />;
  const current =
    items.find((i) => i.to === location.pathname)?.label ?? "Experiment detail";
  const navigation = (
    <>
      <Brand />
      <div className="nav-group-label">RESEARCH WORKSPACE</div>
      <nav aria-label="Main navigation">
        {items.map(({ to, label, icon: Icon }, i) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/app"}
            onClick={() => setMenu(false)}
            className={({ isActive }) =>
              `nav-item ${isActive ? "active" : ""} ${i === 6 ? "nav-divider" : ""}`
            }
          >
            <Icon size={18} />
            <span>{label}</span>
            {to === location.pathname && <span className="nav-active-dot" />}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="research-edition">
          <ShieldCheck size={20} />
          <strong>Built for research</strong>
          <p>One workspace. Every insight traceable.</p>
          <Link to="/" onClick={() => setMenu(false)}>
            About TA-EFL <ArrowUpRight size={14} />
          </Link>
        </div>
        <button
          className="user-profile"
          onClick={async () => {
            await post("/auth/logout");
            qc.clear();
            navigate("/login");
          }}
        >
          <span className="avatar">
            {auth.data.username?.slice(0, 2).toUpperCase()}
          </span>
          <span>
            <strong>{auth.data.username}</strong>
            <small>Researcher · Sign out</small>
          </span>
          <LogOut size={16} />
        </button>
      </div>
    </>
  );
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside className="sidebar">{navigation}</aside>
      <Modal
        open={menu}
        onOpenChange={setMenu}
        title="Workspace navigation"
        description="Navigate the research platform."
      >
        <div className="mobile-navigation">{navigation}</div>
      </Modal>
      <div className="app-body">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button menu-button"
              onClick={() => setMenu(true)}
              aria-label="Open navigation"
            >
              <Menu size={20} />
            </button>
            <span className="workspace-label">Workspace</span>
            <ChevronRight size={13} />
            <strong>{current}</strong>
          </div>
          <div className="topbar-actions">
            <ThemeToggle />
            <Link
              to="/app/history"
              className="search-link"
              aria-label="Find an experiment"
            >
              <Search size={16} />
              <span>Find an experiment</span>
            </Link>
            <span className={`connection ${health.error ? "offline" : ""}`}>
              <Activity size={14} />
              {health.error ? "API unavailable" : "Local research engine"}
            </span>
          </div>
        </header>
        {health.error && (
          <div className="offline-banner" role="alert">
            Connection lost. Previously loaded results may be stale; live status
            cannot be confirmed.
          </div>
        )}
        <main id="main-content" className="main-content">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={reduced ? false : { opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduced ? {} : { opacity: 0 }}
              transition={{ duration: 0.16 }}
            >
              <Suspense fallback={<Loading />}>
                <Outlet />
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </main>
        <footer className="app-footer">
          <span>
            TA-EFL Health <span className="footer-divider">/</span> Research
            edition
          </span>
          <span>Research outputs · not for clinical diagnosis</span>
        </footer>
      </div>
    </div>
  );
}
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/app" element={<Shell />}>
        <Route index element={<Dashboard />} />
        <Route path="simulation" element={<Simulation />} />
        <Route path="network" element={<Simulation networkOnly />} />
        <Route path="trust" element={<Trust />} />
        <Route path="prediction" element={<Prediction />} />
        <Route path="personalization" element={<Personalization />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="experiments/:id" element={<ExperimentDetail />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
