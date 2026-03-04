import Navbar from "@/components/Navbar";
import TradingChart from "@/components/TradingChart";
import NewsFeed from "@/components/NewsFeed";
import AIChatbot from "@/components/AIChatbot";

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <Navbar />

      <main className="flex-1 p-4 flex flex-col lg:flex-row gap-4">
        {/* Left Column: Main Chart */}
        <div className="flex-1 lg:w-2/3 flex flex-col gap-4">
          <TradingChart />
        </div>

        {/* Right Column: News & Chatbot */}
        <div className="w-full lg:w-1/3 flex flex-col gap-4">
          <NewsFeed ticker="FPT" />
          <AIChatbot />
        </div>
      </main>
    </div>
  );
}
