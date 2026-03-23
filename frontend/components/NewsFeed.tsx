"use client";

import { useEffect, useState, useCallback } from "react";
import { Newspaper, RefreshCw, ExternalLink } from "lucide-react";

interface Article {
    title: string;
    url: string;
    source: string;
    score: number;
    sentiment: "positive" | "negative" | "neutral";
}

const SOURCE_COLOR: Record<string, string> = {
    VnExpress:   "text-blue-400  border-blue-700",
    CafeF:       "text-orange-400 border-orange-700",
    TinNhanhCK:  "text-purple-400 border-purple-700",
};

const SENTIMENT_STYLE: Record<string, string> = {
    positive: "bg-green-900/30 text-green-400",
    negative: "bg-red-900/30  text-red-400",
    neutral:  "bg-gray-700    text-gray-300",
};

export default function NewsFeed({ ticker = "" }: { ticker?: string }) {
    const [articles, setArticles] = useState<Article[]>([]);
    const [loading,  setLoading]  = useState(true);
    const [lastFetch, setLastFetch] = useState<Date | null>(null);

    const fetchNews = useCallback(async () => {
        setLoading(true);
        try {
            const param = ticker ? `?symbol=${ticker}&limit=15` : "?limit=15";
            const res   = await fetch(`/api/news/live${param}`);
            if (!res.ok) throw new Error(res.statusText);
            const data  = await res.json();
            setArticles(data.articles ?? []);
            setLastFetch(new Date());
        } catch {
            // keep stale data on error, just stop spinner
        } finally {
            setLoading(false);
        }
    }, [ticker]);

    // Fetch on mount and whenever ticker changes
    useEffect(() => {
        fetchNews();
    }, [fetchNews]);

    // Auto-refresh every 5 minutes
    useEffect(() => {
        const id = setInterval(fetchNews, 5 * 60 * 1000);
        return () => clearInterval(id);
    }, [fetchNews]);

    return (
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg h-[28%] w-full shrink-0 flex flex-col">
            {/* Header */}
            <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between bg-gray-800/50 rounded-t-lg">
                <div className="flex items-center space-x-2">
                    <Newspaper className="w-5 h-5 text-blue-400" />
                    <h2 className="text-white font-semibold">Context-Aware News</h2>
                    {ticker && (
                        <span className="text-xs text-gray-400 bg-gray-700 px-2 py-0.5 rounded">
                            {ticker}
                        </span>
                    )}
                </div>
                <div className="flex items-center space-x-2">
                    {lastFetch && (
                        <span className="text-[10px] text-gray-500">
                            {lastFetch.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}
                        </span>
                    )}
                    <button
                        onClick={fetchNews}
                        disabled={loading}
                        className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
                        title="Refresh news"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                    </button>
                </div>
            </div>

            {/* Feed */}
            <div className="flex-1 overflow-y-auto p-2">
                {loading && articles.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                        Fetching latest headlines...
                    </div>
                ) : articles.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                        No news available
                    </div>
                ) : (
                    articles.map((article, i) => (
                        <div
                            key={i}
                            className="p-3 mb-1.5 border-b border-gray-800 last:border-0 hover:bg-gray-800/50 transition-colors rounded"
                        >
                            <div className="flex justify-between items-center mb-1 gap-2">
                                <div className="flex items-center gap-1.5 flex-wrap">
                                    <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-sm ${SENTIMENT_STYLE[article.sentiment]}`}>
                                        {article.sentiment}
                                    </span>
                                    <span className={`text-[10px] px-1.5 py-0.5 rounded-sm border ${SOURCE_COLOR[article.source] ?? "text-gray-400 border-gray-600"}`}>
                                        {article.source}
                                    </span>
                                </div>
                                <span className="text-[10px] text-gray-500 shrink-0">
                                    {article.score > 0 ? "+" : ""}{article.score.toFixed(2)}
                                </span>
                            </div>
                            <a
                                href={article.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-sm text-gray-200 leading-snug hover:text-white flex items-start gap-1 group"
                            >
                                <span className="flex-1">{article.title}</span>
                                <ExternalLink className="w-3 h-3 mt-0.5 shrink-0 opacity-0 group-hover:opacity-60 transition-opacity" />
                            </a>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
