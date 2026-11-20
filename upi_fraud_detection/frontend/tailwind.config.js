/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#070b14',
          900: '#0a0f1d',
          850: '#0d1527',
          800: '#0f172a',
          700: '#1e293b',
          600: '#334155',
        },
        charcoal: {
          900: '#111927',
          850: '#141e30',
          800: '#162032',
          700: '#1e293b',
        },
        emerald: {
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
        },
        accent: {
          amber: '#f59e0b',
          red: '#ef4444',
          blue: '#3b82f6',
          cyan: '#06b6d4',
          purple: '#8b5cf6',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
