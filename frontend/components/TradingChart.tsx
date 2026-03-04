"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, ISeriesApi } from "lightweight-charts";

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
            height: 500,
        });

        const candlestickSeries = chart.addCandlestickSeries({
            upColor: "#10B981",
            downColor: "#EF4444",
            wickUpColor: "#10B981",
            wickDownColor: "#EF4444",
        });

        // Mock realistic OHLC data
        const mockData = Array.from({ length: 100 }, (_, i) => {
            const date = new Date();
            date.setDate(date.getDate() - (100 - i));
            const open = 100 + Math.random() * 10 - 5;
            const close = open + Math.random() * 10 - 5;
            const high = Math.max(open, close) + Math.random() * 5;
            const low = Math.min(open, close) - Math.random() * 5;

            return {
                time: date.toISOString().split("T")[0],
                open,
                high,
                low,
                close,
            };
        });

        candlestickSeries.setData(mockData);

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
        <div className="w-full h-[500px] bg-gray-900 border border-gray-800 rounded-lg overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-gray-800 flex justify-between items-center">
                <h2 className="text-white font-semibold">Price Chart (Mock)</h2>
                <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">lightweight-charts</span>
            </div>
            <div ref={chartContainerRef} className="flex-1 w-full" />
        </div>
    );
}
