import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    colors: {
      transparent: "transparent",
      current: "currentColor",
      white: "#ffffff",
      black: "#000000",
      slate: {
        50: "#f8fafc",
        100: "#f1f5f9",
        200: "#e2e8f0",
        300: "#c8d3df",
        400: "#aebdca",
        500: "#91a2b3",
        600: "#748596",
        700: "#415061",
        800: "#263241",
        900: "#151e29",
        950: "#0a1017",
      },
      blue: {
        300: "#9bc9ff",
        400: "#69b0ff",
        500: "#3d95ff",
        600: "#2578df",
        950: "#092344",
      },
      indigo: {
        300: "#b9bbff",
        400: "#9ca0ff",
        500: "#7d82f0",
        600: "#656bd4",
      },
      emerald: {
        300: "#80e6c0",
        400: "#4fddb1",
        500: "#2abc8d",
        600: "#1d936d",
      },
      amber: {
        300: "#ffd481",
        400: "#f2b14e",
        500: "#d28f25",
        600: "#ad7018",
      },
      red: {
        300: "#ffb0b0",
        400: "#ff8181",
        500: "#ed5d5d",
        600: "#cc4444",
      },
    },
    extend: {},
  },
  plugins: [],
};

export default config;
