"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, ISeriesApi, LineStyle } from "lightweight-charts";

export default function TradingChart() {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

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

        // 1. Generate realistic synthetic OHLC data
        const mockData = [];
        let currentPrice = 100;
        const now = new Date();

        for (let i = 200; i >= 0; i--) {
            const date = new Date(now);
            date.setDate(date.getDate() - i);

            // Generate some structured swings
            const trend = Math.sin(i / 10) * 5;
            const open = currentPrice;
            const close = open + trend + (Math.random() * 4 - 2);
            const high = Math.max(open, close) + Math.random() * 3;
            const low = Math.min(open, close) - Math.random() * 3;
            currentPrice = close;

            mockData.push({
                time: date.toISOString().split("T")[0],
                open, high, low, close
            });
        }
        candlestickSeries.setData(mockData);

        // 2. Heuristic SMC Logic (Replicated in TS for UI Demo)
        const lookbackWindow = 5;
        const swingHighs = [];
        const swingLows = [];
        const markers: any[] = [];

        // Find Swings
        for (let i = lookbackWindow; i < mockData.length - lookbackWindow; i++) {
            let isHigh = true;
            let isLow = true;
            for (let j = i - lookbackWindow; j <= i + lookbackWindow; j++) {
                if (mockData[j].high > mockData[i].high) isHigh = false;
                if (mockData[j].low < mockData[i].low) isLow = false;
            }
            if (isHigh) swingHighs.push({ idx: i, price: mockData[i].high, time: mockData[i].time });
            if (isLow) swingLows.push({ idx: i, price: mockData[i].low, time: mockData[i].time });
        }

        // 3. Mark BOS / CHoCH (Price Lines for Visuals)
        let trend = 0;
        for (let i = 1; i < swingHighs.length; i++) {
            if (swingHighs[i].price > swingHighs[i - 1].price) {
                const label = trend === 1 ? "BOS" : "CHoCH";
                trend = 1;
                candlestickSeries.createPriceLine({
                    price: swingHighs[i - 1].price,
                    color: '#3b82f6',
                    lineWidth: 2,
                    lineStyle: LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: `Bullish ${label}`,
                });
            }
        }
        for (let i = 1; i < swingLows.length; i++) {
            if (swingLows[i].price < swingLows[i - 1].price) {
                const label = trend === -1 ? "BOS" : "CHoCH";
                trend = -1;
                candlestickSeries.createPriceLine({
                    price: swingLows[i - 1].price,
                    color: '#ef4444',
                    lineWidth: 2,
                    lineStyle: LineStyle.Dotted,
                    axisLabelVisible: true,
                    title: `Bearish ${label}`,
                });
            }
        }

        // 4. Mark AI Execution Signals (Liquidity Sweeps)
        // We inject mock LSTM signals where anomalous deep wicks occur
        for (let i = 2; i < mockData.length; i++) {
            const candle = mockData[i];
            const wickRatio = (Math.min(candle.open, candle.close) - candle.low) / (candle.high - candle.low + 0.01);

            if (wickRatio > 0.65 && candle.close > candle.open) {
                markers.push({
                    time: candle.time,
                    position: 'belowBar',
                    color: '#10B981',
                    shape: 'arrowUp',
                    text: 'AI BUY (Sweep)',
                    size: 2
                });
            }
        }
        candlestickSeries.setMarkers(markers);

        chartRef.current = chart;
        seriesRef.current = candlestickSeries;

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
    }, []);

    return (
        <div className="w-full h-[600px] bg-gray-900 border border-gray-800 rounded-lg overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-gray-800 flex justify-between items-center">
                <div className="flex items-center space-x-4">
                    <h2 className="text-white font-semibold">Green Dragon Heuristic UI</h2>
                    <span className="text-xs font-bold text-blue-400 bg-blue-900/30 px-2 py-1 rounded border border-blue-800">BOS / CHoCH</span>
                    <span className="text-xs font-bold text-green-400 bg-green-900/30 px-2 py-1 rounded border border-green-800">AI Signals</span>
                </div>
                <span className="text-xs text-gray-500">Heuristic Overlay Mode Active</span>
            </div>
            <div ref={chartContainerRef} className="flex-1 w-full relative" />
        </div>
    );
}
