import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.TA_EFL_API_TARGET || "http://127.0.0.1:8780",
        changeOrigin: false,
      },
    },
  },
  build: { chunkSizeWarningLimit: 1100 },
});
