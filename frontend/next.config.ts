import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const backend = (process.env.API_BACKEND_URL ||
      (process.env.NODE_ENV === "production"
        ? "https://lck-win-predictor-production.up.railway.app"
        : "http://127.0.0.1:8000")).replace(/\/$/, "");
    return [{ source: "/api/:path*", destination: `${backend}/:path*` }];
  },
};

export default nextConfig;
