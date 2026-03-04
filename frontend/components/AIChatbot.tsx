"use client";

import { useState } from "react";
import { Send, Bot, User } from "lucide-react";

export default function AIChatbot() {
    const [messages, setMessages] = useState<{ role: "bot" | "user"; text: string }[]>([
        { role: "bot", text: "Hello! I am your Financial Copilot. Ask me about SMC concepts, indicators, or market liquidity." },
    ]);
    const [input, setInput] = useState("");

    const handleSend = () => {
        if (!input.trim()) return;

        const userMessage = input.trim();
        setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
        setInput("");

        // Mock AI Responses based on keywords
        setTimeout(() => {
            const lower = userMessage.toLowerCase();
            let botResponse = "I am a mock AI Copilot. Connect me back to the Python FastAPI backend to get real analysis!";

            if (lower.includes("liquidity")) {
                botResponse = "Liquidity refers to areas on a chart where a large number of stop-loss orders are placed. In Smart Money Concepts (SMC), institutional traders target these areas (Liquidity Sweeps) to fill their large positions.";
            } else if (lower.includes("order block") || lower.includes("ob")) {
                botResponse = "An Order Block (OB) is the last bearish candle before a strong bullish move, or the last bullish candle before a strong bearish move. It represents the footprint of institutional order packing.";
            }

            setMessages((prev) => [...prev, { role: "bot", text: botResponse }]);
        }, 600);
    };

    return (
        <div className="bg-gray-900 border border-gray-800 rounded-lg flex flex-col h-[400px]">
            <div className="px-4 py-3 border-b border-gray-800 flex justify-between items-center bg-gray-800/50 rounded-t-lg">
                <div className="flex items-center space-x-2">
                    <Bot className="w-5 h-5 text-purple-400" />
                    <h2 className="text-white font-semibold">Financial Copilot</h2>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div className={`flex items-start max-w-[85%] space-x-2 ${msg.role === "user" ? "flex-row-reverse space-x-reverse" : "flex-row"}`}>
                            <div className={`p-1.5 rounded-full flex-shrink-0 ${msg.role === "user" ? "bg-blue-600" : "bg-purple-900"}`}>
                                {msg.role === "user" ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-purple-200" />}
                            </div>
                            <div className={`text-sm px-3 py-2 rounded-lg ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-200"}`}>
                                {msg.text}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <div className="p-3 border-t border-gray-800">
                <div className="relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                        placeholder="Ask about SMC or markets..."
                        className="w-full bg-gray-800 border border-gray-700 text-sm text-white rounded-md pl-3 pr-10 py-2 focus:outline-none focus:border-purple-500"
                    />
                    <button
                        onClick={handleSend}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-white transition-colors"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
}
