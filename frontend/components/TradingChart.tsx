"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, IChartApi, ISeriesApi, LineStyle, MouseEventParams } from "lightweight-charts";

interface TradingChartProps {
    onSignalClick?: (context: any) => void;
}

export default function TradingChart({ onSignalClick }: TradingChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const dataRef = useRef<any>(null); // Store fetched data for click handling
    const [ticker, setTicker] = useState("FPT");
    const [days, setDays] = useState(200);
    const [loading, setLoading] = useState(true);
    const [dateRange, setDateRange] = useState({ start: "", end: "" });

    const [tooltipData, setTooltipData] = useState<{
        visible: boolean;
        date: string;
        open: string;
        high: string;
        low: string;
        close: string;
        volume: string;
        signal: string;
    } | null>(null);

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
                const response = await fetch(`http://localhost:8000/api/v1/chart-data/${ticker}?days=${days}`);
                if (!response.ok) throw new Error("Network response was not ok");
                const data = await response.json();
                dataRef.current = data;

                // 1. Set OHLCV Data
                candlestickSeries.setData(data.ohlcv);

                // Extract dynamic date range
                if (data.ohlcv.length > 0) {
                    setDateRange({
                        start: data.ohlcv[0].time,
                        end: data.ohlcv[data.ohlcv.length - 1].time
                    });
                }

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

                // 3. Mark BOS / CHoCH (Finite Price Lines)
                // Filter to only the 5 most recent to avoid extreme clutter
                const recentBOS = data.smc.bos.slice(-5);
                const recentCHoCH = data.smc.choch.slice(-5);

                const renderFiniteLine = (marker: any, isBOS: boolean) => {
                    const lineSeries = chart.addLineSeries({
                        color: marker.direction === 'bullish' ? '#3b82f6' : '#ef4444',
                        lineWidth: 2,
                        lineStyle: isBOS ? LineStyle.Dashed : LineStyle.Dotted,
                        crosshairMarkerVisible: false,
                        lastValueVisible: false,
                        priceLineVisible: false,
                    });
                    lineSeries.setData([
                        { time: marker.date_start, value: marker.price },
                        { time: marker.date_end, value: marker.price }
                    ]);
                };

                recentBOS.forEach((marker: any) => renderFiniteLine(marker, true));
                recentCHoCH.forEach((marker: any) => renderFiniteLine(marker, false));

                // Automatically fit content
                chart.timeScale().fitContent();

            } catch (error) {
                console.error("Error fetching chart data:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();

        const handleChartClick = (param: MouseEventParams) => {
            if (!param || !param.time || !dataRef.current || !onSignalClick) return;

            const clickedTime = param.time as string;
            const data = dataRef.current;

            // Find nearest signal within a 3-day window for robust clicking UX
            if (!data.action_signals || data.action_signals.length === 0) return;

            let closestSignal = null;
            let minDiff = Infinity;

            for (const sig of data.action_signals) {
                const diff = Math.abs(new Date(sig.time).getTime() - new Date(clickedTime).getTime());
                if (diff < minDiff) {
                    minDiff = diff;
                    closestSignal = sig;
                }
            }

            // 3 days in milliseconds = 3 * 24 * 60 * 60 * 1000 = 259200000
            if (closestSignal && minDiff <= 259200000) {
                // Determine SMC context around this date
                const nearbyBOS = data.smc.bos.find((b: any) => b.date_start <= clickedTime && b.date_end >= clickedTime)
                    ? "Structural BOS confirmed."
                    : "";

                const smc_context = nearbyBOS || "Deep liquidity sweep detected strictly below local momentum.";

                onSignalClick({
                    ticker: data.ticker,
                    date: closestSignal.time, // Exact signal date
                    price: closestSignal.price,
                    score: closestSignal.score,
                    smc_context: smc_context,
                    _t: Date.now() // Ensure context trigger runs again
                });
            }
        };

        chart.subscribeClick(handleChartClick);

        const handleCrosshairMove = (param: MouseEventParams) => {
            if (!param.point || !param.time || param.point.x < 0 || param.point.x > chartContainerRef.current!.clientWidth || param.point.y < 0 || param.point.y > chartContainerRef.current!.clientHeight) {
                setTooltipData(null);
                return;
            }

            const data = dataRef.current;
            if (!data) return;

            const timeStr = param.time as string;
            const barData: any = param.seriesData.get(seriesRef.current!);

            if (!barData) {
                setTooltipData(null);
                return;
            }

            // Find matching SMC labels at this time
            let labels = [];

            const matchMarker = (marker: any) => marker.date_start <= timeStr && marker.date_end >= timeStr;
            const isBOS = data.smc.bos.some(matchMarker);
            if (isBOS) labels.push("BOS");

            const isCHoCH = data.smc.choch.some(matchMarker);
            if (isCHoCH) labels.push("CHoCH");

            // Limit order blocks search if array exists
            if (data.smc.order_blocks && data.smc.order_blocks.length > 0) {
                const isOB = data.smc.order_blocks.some((marker: any) => marker.start_date <= timeStr && marker.end_date >= timeStr);
                if (isOB) labels.push("Order Block");
            }

            const hoveredSig = data.action_signals.find((s: any) => s.time === timeStr);
            if (hoveredSig) labels.push(`AI BUY (${hoveredSig.score.toFixed(2)})`);

            setTooltipData({
                visible: true,
                date: timeStr,
                open: barData.open.toLocaleString(),
                high: barData.high.toLocaleString(),
                low: barData.low.toLocaleString(),
                close: barData.close.toLocaleString(),
                volume: data.ohlcv.find((c: any) => c.time === timeStr)?.volume?.toLocaleString() || "N/A",
                signal: labels.length > 0 ? labels.join(" | ") : "None"
            });
        };

        chart.subscribeCrosshairMove(handleCrosshairMove);

        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            chart.unsubscribeClick(handleChartClick);
            chart.unsubscribeCrosshairMove(handleCrosshairMove);
            chart.remove();
        };
    }, [ticker, days]);

    return (
        <div className="w-full h-[600px] bg-gray-900 border border-gray-800 rounded-lg overflow-hidden flex flex-col relative" style={{ userSelect: "none" }}>
            <div className="px-4 py-3 border-b border-gray-800 z-10 bg-gray-900 overflow-x-auto scrollbar-hide">
                <div className="flex flex-row items-center justify-between w-full gap-4 p-2">
                    <div className="flex items-center space-x-4 flex-shrink-0">
                        <h2 className="text-white font-semibold whitespace-nowrap">Green Dragon Live Feed</h2>
                        <select
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value)}
                            className="bg-gray-800 text-white text-sm px-3 py-1 rounded border border-gray-700 outline-none focus:border-green-500 cursor-pointer flex-shrink-0"
                        >
                            <option value="FPT">FPT</option>
                            <option value="MBB">MBB</option>
                            <option value="SSI">SSI</option>
                            <option value="HPG">HPG</option>
                            <option value="VNM">VNM</option>
                            <option value="MWG">MWG</option>
                        </select>

                        {/* Timeframe Buttons */}
                        <div className="flex space-x-1 border border-gray-700 rounded overflow-hidden flex-shrink-0">
                            {[
                                { label: '7D', val: 7 },
                                { label: '1M', val: 30 },
                                { label: '3M', val: 90 },
                                { label: '1Y', val: 200 }
                            ].map(tf => (
                                <button
                                    key={tf.label}
                                    onClick={() => setDays(tf.val)}
                                    className={`px-3 py-1 text-xs font-semibold transition-colors ${days === tf.val ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
                                >
                                    {tf.label}
                                </button>
                            ))}
                        </div>

                        {/* Dynamic Date Range Indicator */}
                        {dateRange.start && dateRange.end && (
                            <div className="flex items-center space-x-2 text-xs text-gray-400 bg-gray-800/50 px-3 py-1.5 rounded border border-gray-700 whitespace-nowrap flex-shrink-0">
                                <span>Showing Period:</span>
                                <span className="text-gray-200 font-medium">{dateRange.start}</span>
                                <span>to</span>
                                <span className="text-gray-200 font-medium">{dateRange.end}</span>
                            </div>
                        )}

                        <span className="text-xs font-bold text-blue-400 bg-blue-900/30 px-2 py-1 rounded border border-blue-800 hidden xl:inline flex-shrink-0 whitespace-nowrap">Finite SMC</span>
                        <span className="text-xs font-bold text-green-400 bg-green-900/30 px-2 py-1 rounded border border-green-800 hidden xl:inline flex-shrink-0 whitespace-nowrap">LSTM Act 0.635</span>
                    </div>

                    <div className="flex items-center space-x-2 flex-shrink-0">
                        {loading && <span className="text-xs text-green-400 animate-pulse whitespace-nowrap">Syncing Engine...</span>}
                        <span className="text-xs text-gray-500 whitespace-nowrap">Real-time Optuna Evaluator Active</span>
                    </div>
                </div>
            </div>
            <div className="flex-1 w-full relative">
                <div ref={chartContainerRef} className="absolute inset-0" />

                {/* Hover Crosshair Legend Tooltip */}
                {tooltipData && tooltipData.visible && (
                    <div className="absolute top-4 left-4 z-50 flex flex-col gap-1.5 p-3 rounded-md bg-gray-900/85 backdrop-blur-sm border border-gray-700 pointer-events-none shadow-lg whitespace-nowrap">
                        <div className="text-sm font-semibold text-gray-100">Date: {tooltipData.date}</div>
                        <div className="flex flex-row items-center gap-4 text-xs font-mono text-gray-300">
                            <span><span className="text-gray-500">O:</span> {tooltipData.open}</span>
                            <span><span className="text-gray-500">H:</span> {tooltipData.high}</span>
                            <span><span className="text-gray-500">L:</span> {tooltipData.low}</span>
                            <span><span className="text-gray-500">C:</span> {tooltipData.close}</span>
                            <span><span className="text-gray-500">Vol:</span> {tooltipData.volume}</span>
                        </div>
                        {tooltipData.signal !== "None" && (
                            <div className={`text-sm font-bold mt-1 ${tooltipData.signal.includes('BUY') || tooltipData.signal.includes('BOS') ? 'text-green-400' : 'text-red-400'}`}>
                                Signal: {tooltipData.signal}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
