"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, IChartApi, ISeriesApi, LineStyle } from "lightweight-charts";

export default function TradingChart() {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const [ticker, setTicker] = useState("FPT");
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: "#111827" },
                textColor: "#9CA3AF",
            },
            grid: {
                vertLines: { color: "#1F2937" },
                horzLines: { color: "#1F2937" },
            },
            width: chartContainerRef.current.clientWidth,
            height: 600,
        });

        const candlestickSeries = chart.addCandlestickSeries({
            upColor: "#10B981",
            downColor: "#EF4444",
            wickUpColor: "#10B981",
            wickDownColor: "#EF4444",
        });

        chartRef.current = chart;
        seriesRef.current = candlestickSeries;

        const fetchData = async () => {
            setLoading(true);
            try {
                // Fetch real data from FastAPI backend
                const response = await fetch(`http://localhost:8000/api/v1/chart-data/${ticker}?days=200`);
                if (!response.ok) throw new Error("Network response was not ok");
                const data = await response.json();

                // 1. Set OHLCV Data
                candlestickSeries.setData(data.ohlcv);

                const markers: any[] = [];

                // 2. Mark AI Execution Signals (Liquidity Sweeps)
                data.action_signals.forEach((signal: any) => {
                    markers.push({
                        time: signal.time,
                        position: 'belowBar',
                        color: '#10B981',
                        shape: 'arrowUp',
                        text: `AI BUY (${signal.score.toFixed(2)})`,
                        size: 2
                    });
                });
                candlestickSeries.setMarkers(markers);

                // 3. Mark BOS / CHoCH (Price Lines)
                // BOS Lines
                data.smc.bos.forEach((marker: any) => {
                    candlestickSeries.createPriceLine({
                        price: marker.price,
                        color: marker.direction === 'bullish' ? '#3b82f6' : '#ef4444',
                        lineWidth: 2,
                        lineStyle: LineStyle.Dashed,
                        axisLabelVisible: true,
                        title: `${marker.direction === 'bullish' ? 'Bull' : 'Bear'} BOS`,
                    });
                });

                // CHoCH Lines
                data.smc.choch.forEach((marker: any) => {
                    candlestickSeries.createPriceLine({
                        price: marker.price,
                        color: marker.direction === 'bullish' ? '#3b82f6' : '#ef4444',
                        lineWidth: 2,
                        lineStyle: LineStyle.Dotted,
                        axisLabelVisible: true,
                        title: `${marker.direction === 'bullish' ? 'Bull' : 'Bear'} CHoCH`,
                    });
                });

                // Automatically fit content
                chart.timeScale().fitContent();

            } catch (error) {
                console.error("Error fetching chart data:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();

        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            chart.remove();
        };
    }, [ticker]);

    return (
        <div className="w-full h-[600px] bg-gray-900 border border-gray-800 rounded-lg overflow-hidden flex flex-col relative">
            <div className="px-4 py-3 border-b border-gray-800 flex justify-between items-center z-10 bg-gray-900">
                <div className="flex items-center space-x-4">
                    <h2 className="text-white font-semibold">Green Dragon Live Feed</h2>
                    <select
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value)}
                        className="bg-gray-800 text-white text-sm px-3 py-1 rounded border border-gray-700 outline-none focus:border-green-500"
                    >
                        <option value="FPT">FPT</option>
                        <option value="MBB">MBB</option>
                        <option value="SSI">SSI</option>
                        <option value="HPG">HPG</option>
                        <option value="VNM">VNM</option>
                        <option value="MWG">MWG</option>
                    </select>
                    <span className="text-xs font-bold text-blue-400 bg-blue-900/30 px-2 py-1 rounded border border-blue-800">SMC (BOS/CHoCH)</span>
                    <span className="text-xs font-bold text-green-400 bg-green-900/30 px-2 py-1 rounded border border-green-800">LSTM Action (Threshold 0.635)</span>
                </div>
                <div className="flex items-center space-x-2">
                    {loading && <span className="text-xs text-green-400 animate-pulse">Syncing Engine...</span>}
                    <span className="text-xs text-gray-500">Real-time Optuna Evaluator Active</span>
                </div>
            </div>
            <div ref={chartContainerRef} className="flex-1 w-full relative" />
        </div>
    );
}
