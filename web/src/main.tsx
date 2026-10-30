import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "motion/react";
import "@fontsource-variable/dm-sans";
import "@fontsource-variable/manrope";
import "./styles.css";
import App from "./app";
import { ThemeProvider } from "./theme";
const client = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 10000, refetchOnWindowFocus: true, retry: 1 },
    mutations: { retry: false },
  },
});
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <main className="loading">
        <h1>The workspace could not be displayed.</h1>
        <button className="button primary" onClick={() => location.reload()}>
          Reload workspace
        </button>
      </main>
    ) : (
      this.props.children
    );
  }
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <ErrorBoundary>
        <QueryClientProvider client={client}>
          <MotionConfig reducedMotion="user">
            <BrowserRouter>
              <App />
            </BrowserRouter>
          </MotionConfig>
        </QueryClientProvider>
      </ErrorBoundary>
    </ThemeProvider>
  </React.StrictMode>,
);
