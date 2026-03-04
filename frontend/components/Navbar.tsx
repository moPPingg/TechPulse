"use client";

import { useState } from "react";
import { TrendingUp, Clock, Activity } from "lucide-react";

export default function Navbar() {
  const [ticker, setTicker] = useState("FPT");
  const [timeframe, setTimeframe] = useState("1D");

  return (
    <nav className="flex items-center justify-between p-4 bg-gray-900 border-b border-gray-800 text-white">
      <div className="flex items-center space-x-2">
        <TrendingUp className="text-green-500" />
        <span className="text-xl font-bold tracking-tight">Green Dragon Core</span>
      </div>

      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-gray-400" />
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="bg-gray-800 border border-gray-700 text-sm rounded-md px-3 py-1.5 focus:outline-none focus:border-green-500"
          >
            <option value="FPT">FPT</option>
            <option value="HPG">HPG</option>
            <option value="VCB">VCB</option>
            <option value="SSI">SSI</option>
          </select>
        </div>

        <div className="flex items-center bg-gray-800 rounded-md p-1">
          <Clock className="w-4 h-4 text-gray-400 ml-2 mr-1" />
          {["1D", "7D", "1M", "3M"].map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1 text-xs font-medium rounded-sm transition-colors ${
                timeframe === tf ? "bg-gray-600 text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
}
