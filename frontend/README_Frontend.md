# Green Dragon - Frontend Dashboard

This directory contains the new, clean real-time web interface for the Green Dragon Quantitative Trading System, built as a TradingView Clone dashboard.

## Tech Stack
- Frontend Framework: Next.js 15 (App Router) + TypeScript
- Styling: Tailwind CSS
- UI Icons: `lucide-react`
- Charting Engine: `lightweight-charts` (TradingView)

## How to Run

1. Open a new terminal and navigate to this folder:
   ```bash
   cd frontend
   ```

2. (Optional) If not already installed, install the Node modules:
   ```bash
   npm install
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Connecting to the Python FastAPI Backend
Presently, the dashboard uses mocked (hardcoded) data to demonstrate its capabilities. To feed it real data:

1. **API Endpoints**: Open `api.py` in the root repository and ensure the Python FastAPI server has standard REST/WebSocket endpoints returning JSON. For example, `/api/stock/{ticker}/ohlcv` or `/api/stock/{ticker}/news`.
2. **CORS Policy**: Make sure FastAPI's CORS middleware allows requests from `http://localhost:3000`.
3. **Fetching Data**: Replace the mocked `setTimeout` or hardcoded arrays inside the React components (`TradingChart.tsx`, `NewsFeed.tsx`, `AIChatbot.tsx`) with dynamic API calls using the native `fetch` API hooked securely with React's `useEffect` or React Query.
