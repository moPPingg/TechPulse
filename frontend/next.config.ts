import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",   // static HTML export — output goes to frontend/out/
  trailingSlash: true,
};

export default nextConfig;
