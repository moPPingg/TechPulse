"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import TradingChart from "@/components/TradingChart";
import NewsFeed from "@/components/NewsFeed";
import ChatbotPanel from "@/components/ChatbotPanel";

export default function Home() {
  const [chatContext, setChatContext] = useState<any>(null);

  return (
    <div className="flex flex-col xl:flex-row h-screen w-screen overflow-hidden p-3 gap-3 bg-[#0B0E14] text-white">
      {/* Left Column: Main Chart & News (70%) */}
      <div className="flex flex-col w-full xl:w-[70%] h-full gap-3 overflow-hidden">
        <TradingChart onSignalClick={setChatContext} />
        <NewsFeed ticker="FPT" />
      </div>

      {/* Right Column: Chatbot (30%) */}
      <div className="w-full xl:w-[30%] h-full shrink-0 flex flex-col rounded-lg border border-gray-800 bg-gray-900/50 overflow-hidden">
        <ChatbotPanel activeContext={chatContext} />
      </div>
    </div>
  );
}
