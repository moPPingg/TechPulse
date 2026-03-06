"use client";

import { Newspaper } from "lucide-react";

export default function NewsFeed({ ticker = "FPT" }: { ticker?: string }) {
    // Mock logic based on ticker
    const mockNews = [
        {
            id: 1,
            title: `${ticker} announces Q3 financial results breaking previous records.`,
            time: "2 hours ago",
            sentiment: "positive",
        },
        {
            id: 2,
            title: `Institutional buying detected around ${ticker} key support levels.`,
            time: "5 hours ago",
            sentiment: "neutral",
        },
        {
            id: 3,
            title: `Macroeconomic data challenges ${ticker === "FPT" ? "Tech sector" : ticker === "HPG" ? "Steel manufacturing" : "the broad market"}.`,
            time: "1 day ago",
            sentiment: "negative",
        }
    ];

    return (
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg h-[28%] w-full shrink-0 flex flex-col">
            <div className="px-4 py-3 border-b border-gray-800 flex items-center space-x-2 bg-gray-800/50 rounded-t-lg">
                <Newspaper className="w-5 h-5 text-blue-400" />
                <h2 className="text-white font-semibold">Context-Aware News</h2>
            </div>

            <div className="flex-1 overflow-y-auto p-2">
                {mockNews.map((news) => (
                    <div key={news.id} className="p-3 mb-2 border-b border-gray-800 last:border-0 hover:bg-gray-800/50 cursor-pointer transition-colors rounded">
                        <div className="flex justify-between items-start mb-1">
                            <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-sm 
                ${news.sentiment === 'positive' ? 'bg-green-900/30 text-green-400' :
                                    news.sentiment === 'negative' ? 'bg-red-900/30 text-red-400' :
                                        'bg-gray-700 text-gray-300'}`}
                            >
                                {news.sentiment}
                            </span>
                            <span className="text-xs text-gray-500">{news.time}</span>
                        </div>
                        <p className="text-sm text-gray-200 leading-snug">{news.title}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
