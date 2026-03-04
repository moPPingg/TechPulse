"use client";

import { useState, useEffect, useRef } from "react";
import { Send, Bot, User, FileText } from "lucide-react";

interface ChatMessage {
    role: "bot" | "user";
    text: string;
}

interface ChatContext {
    ticker: string;
    date: string;
    price: number;
    score: number;
    smc_context: string;
}

interface ChatbotPanelProps {
    activeContext: ChatContext | null;
}

export default function ChatbotPanel({ activeContext }: ChatbotPanelProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([
        { role: "bot", text: "Hello! I am Green Dragon, your Quantitative AI Assistant. Ask me about SMC concepts, indicators, or click on a BUY signal on the chart to hear my execution reasoning." },
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        if (activeContext) {
            handleContextTrigger(activeContext);
        }
    }, [activeContext]);

    const handleContextTrigger = async (context: ChatContext) => {
        setMessages((prev) => [
            ...prev,
            { role: "user", text: `Why did you execute a BUY signal on ${context.ticker} at ${context.date}?` }
        ]);
        await fetchChatResponse(null, context);
    };

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput("");
        setMessages((prev) => [...prev, { role: "user", text: userMessage }]);

        await fetchChatResponse(userMessage, null);
    };

    const fetchChatResponse = async (userText: string | null, context: ChatContext | null) => {
        setIsLoading(true);
        try {
            const apiMessages = messages.map(m => ({ role: m.role, content: m.text }));
            if (userText) {
                apiMessages.push({ role: "user", content: userText });
            } else if (context) {
                apiMessages.push({ role: "user", content: `Why did you execute a BUY signal on ${context.ticker} at ${context.date}?` });
            }

            const payload = {
                messages: apiMessages,
                context: context ? {
                    ticker: context.ticker,
                    date: context.date,
                    price: context.price,
                    score: context.score,
                    smc_context: context.smc_context
                } : null
            };

            const response = await fetch("http://localhost:8000/api/v1/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Failed to fetch chat response");
            const data = await response.json();

            setMessages((prev) => [...prev, { role: "bot", text: data.reply }]);
        } catch (error) {
            console.error(error);
            setMessages((prev) => [...prev, { role: "bot", text: "Connection to Green Dragon AI failed. Ensure the FastAPI backend is running." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="bg-gray-900 border border-gray-800 rounded-lg flex flex-col h-full min-h-[600px] shadow-lg">
            <div className="px-4 py-3 border-b border-gray-800 flex justify-between items-center bg-gray-800/80 rounded-t-lg backdrop-blur shadow-sm z-10">
                <div className="flex items-center space-x-3">
                    <div className="bg-green-500/10 p-2 rounded-full border border-green-500/20">
                        <Bot className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                        <h2 className="text-white font-bold tracking-wide">Green Dragon AI</h2>
                        <span className="text-xs text-green-400 animate-pulse">● Quant Copilot Online</span>
                    </div>
                </div>
                {activeContext && (
                    <div className="flex items-center space-x-1 border border-blue-500/30 bg-blue-500/10 px-2 py-1 rounded text-xs text-blue-300">
                        <FileText className="w-3 h-3" />
                        <span>Chart Context Active</span>
                    </div>
                )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-5 custom-scrollbar bg-gradient-to-b from-gray-900 to-[#0B0F19]">
                {messages.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                        <div className={`flex flex-col max-w-[85%] space-y-1 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                            <div className="flex items-start space-x-2">
                                {msg.role === "bot" && (
                                    <div className="mt-1 p-1.5 rounded-full bg-gray-800 border border-gray-700 flex-shrink-0">
                                        <Bot className="w-4 h-4 text-green-400" />
                                    </div>
                                )}
                                <div
                                    className={`text-sm px-4 py-2.5 shadow-md ${msg.role === "user"
                                        ? "bg-blue-600 text-white rounded-2xl rounded-tr-sm border border-blue-500"
                                        : "bg-gray-800 text-gray-200 rounded-2xl rounded-tl-sm border border-gray-700 whitespace-pre-wrap leading-relaxed"
                                        }`}
                                    dangerouslySetInnerHTML={{
                                        __html: msg.text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
                                    }}
                                />
                                {msg.role === "user" && (
                                    <div className="mt-1 p-1.5 rounded-full bg-blue-600 border border-blue-500 flex-shrink-0">
                                        <User className="w-4 h-4 text-white" />
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex justify-start animate-in fade-in">
                        <div className="flex items-start space-x-2">
                            <div className="mt-1 p-1.5 rounded-full bg-gray-800 border border-gray-700 flex-shrink-0">
                                <Bot className="w-4 h-4 text-green-400" />
                            </div>
                            <div className="bg-gray-800 text-gray-400 text-sm px-4 py-3 rounded-2xl rounded-tl-sm border border-gray-700 flex items-center space-x-1">
                                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></span>
                                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></span>
                                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></span>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="p-4 border-t border-gray-800 bg-gray-900 rounded-b-lg">
                <div className="relative group">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                        placeholder="Message Green Dragon AI..."
                        disabled={isLoading}
                        className="w-full bg-gray-800/80 border border-gray-700 text-sm text-white rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500/50 transition-all placeholder-gray-500 disabled:opacity-50"
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-green-600 hover:bg-green-500 active:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
                <div className="text-center mt-2">
                    <span className="text-[10px] text-gray-500">Green Dragon uses advanced Optuna-tuned heuristics. Responses may take a moment.</span>
                </div>
            </div>
        </div>
    );
}
