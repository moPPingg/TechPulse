"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import TradingChart from "@/components/TradingChart";
import NewsFeed from "@/components/NewsFeed";
import ChatbotPanel from "@/components/ChatbotPanel";

export default function Home() {
  const [chatContext, setChatContext] = useState<any>(null);

  return (
    <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
      <Navbar />

      <main className="flex-1 p-4 flex flex-col lg:flex-row gap-4 overflow-hidden">
        {/* Left Column: Main Chart (70%) */}
        <div className="w-full lg:w-[70%] h-full flex flex-col gap-4 overflow-hidden">
          <TradingChart onSignalClick={setChatContext} />
          <NewsFeed ticker="FPT" />
        </div>

        {/* Right Column: Chatbot (30%) */}
        <div className="w-full lg:w-[30%] h-full flex flex-col gap-4 overflow-hidden">
          <ChatbotPanel activeContext={chatContext} />
        </div>
      </main>
    </div>
  );
}
