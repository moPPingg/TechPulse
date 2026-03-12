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
    const smcMarkersDataRef = useRef<any[]>([]);
    const updateLabelPositionsRef = useRef<(() => void) | null>(null);
    const [smcLabels, setSmcLabels] = useState<{ id: string, x: number, y: number, text: string, color: string, direction: string }[]>([]);
    const [ticker, setTicker] = useState("FPT");
    const [timeframe, setTimeframe] = useState('1Y');
    const [loading, setLoading] = useState(true);
    const [dateRange, setDateRange] = useState({ start: "", end: "" });

    const [tooltip, setTooltip] = useState<{
        visible: boolean;
        x: number;
        y: number;
        data: {
            date: string;
            open: string;
            high: string;
            low: string;
            close: string;
            volume: string;
            signal: string;
        } | null;
    }>({ visible: false, x: 0, y: 0, data: null });

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#D9D9D9',
            },
            rightPriceScale: {
                visible: true,
                autoScale: true,
                alignLabels: true,
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.1,
                },
            },
            timeScale: {
                rightOffset: 12,
                barSpacing: 10,
                timeVisible: true,
            },
            grid: {
                vertLines: { color: "#1F2937" },
                horzLines: { color: "#1F2937" },
            },
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: true,
            },
            handleScale: {
                axisPressedMouseMove: {
                    time: true,
                    price: true,
                },
                mouseWheel: true,
                pinch: true,
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
                const response = await fetch(`http://localhost:8000/api/v1/chart-data/${ticker}?days=3000`);
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

                // 2. Mark AI Execution Signals (Liquidity Sweeps / Take Profit / Stop Loss / Time Exit)
                data.action_signals.forEach((signal: any) => {
                    if (signal.type === 'BUY') {
                        markers.push({
                            time: signal.time,
                            position: 'belowBar',
                            color: '#10B981',
                            shape: 'arrowUp',
                            text: `(${signal.score.toFixed(2)})`,
                            size: 1
                        });
                    } else if (signal.type === 'SELL') {
                        markers.push({
                            time: signal.time,
                            position: 'aboveBar',
                            color: '#EF4444',
                            shape: 'arrowDown',
                            text: `${signal.reason}`,
                            size: 1
                        });
                    }
                });
                // 4. Localized Trade Risk/Reward Brackets
                let currentBuySignal: any = null;
                data.action_signals.forEach((signal: any) => {
                    if (signal.type === 'BUY') {
                        currentBuySignal = signal;
                    } else if (signal.type === 'SELL' && currentBuySignal) {
                        const targetPrice = currentBuySignal.price * 1.03;
                        const stopPrice = currentBuySignal.price * 0.98;
                        
                        const targetSeries = chart.addLineSeries({
                            color: 'rgba(16, 185, 129, 0.8)',
                            lineWidth: 1,
                            lineStyle: LineStyle.Dashed,
                            crosshairMarkerVisible: false,
                            lastValueVisible: false,
                            priceLineVisible: false,
                            autoscaleInfoProvider: () => null,
                        });
                        targetSeries.setData([
                            { time: currentBuySignal.time, value: targetPrice },
                            { time: signal.time, value: targetPrice }
                        ]);

                        const stopSeries = chart.addLineSeries({
                            color: 'rgba(239, 68, 68, 0.8)',
                            lineWidth: 1,
                            lineStyle: LineStyle.Dashed,
                            crosshairMarkerVisible: false,
                            lastValueVisible: false,
                            priceLineVisible: false,
                            autoscaleInfoProvider: () => null,
                        });
                        stopSeries.setData([
                            { time: currentBuySignal.time, value: stopPrice },
                            { time: signal.time, value: stopPrice }
                        ]);

                        currentBuySignal = null;
                    }
                });

                // Draw brackets for an active trade that hasn't closed yet
                if (currentBuySignal) {
                    const lastBar = data.ohlcv[data.ohlcv.length - 1];
                    if (lastBar && lastBar.time >= currentBuySignal.time) {
                        const targetPrice = currentBuySignal.price * 1.03;
                        const stopPrice = currentBuySignal.price * 0.98;

                        const targetSeries = chart.addLineSeries({
                            color: 'rgba(16, 185, 129, 0.8)',
                            lineWidth: 1,
                            lineStyle: LineStyle.Dashed,
                            crosshairMarkerVisible: false,
                            lastValueVisible: false,
                            priceLineVisible: false,
                            autoscaleInfoProvider: () => null,
                        });
                        targetSeries.setData([
                            { time: currentBuySignal.time, value: targetPrice },
                            { time: lastBar.time, value: targetPrice }
                        ]);

                        const stopSeries = chart.addLineSeries({
                            color: 'rgba(239, 68, 68, 0.8)',
                            lineWidth: 1,
                            lineStyle: LineStyle.Dashed,
                            crosshairMarkerVisible: false,
                            lastValueVisible: false,
                            priceLineVisible: false,
                            autoscaleInfoProvider: () => null,
                        });
                        stopSeries.setData([
                            { time: currentBuySignal.time, value: stopPrice },
                            { time: lastBar.time, value: stopPrice }
                        ]);
                    }
                }

                // 3. Mark BOS / CHoCH (Finite Price Lines)
                // Render all historical BOS / CHoCH
                smcMarkersDataRef.current = [];
                const recentBOS = data.smc.bos;
                const recentCHoCH = data.smc.choch;

                const renderFiniteLine = (marker: any, isBOS: boolean, index: number) => {
                    const tvColor = isBOS ? '#9ca3af' : '#facc15'; // Grey for BOS, Yellow for CHOCH

                    const lineSeries = chart.addLineSeries({
                        color: tvColor,
                        lineWidth: 1,
                        lineStyle: LineStyle.Dashed, // 2 is Dashed
                        crosshairMarkerVisible: false,
                        lastValueVisible: false,
                        priceLineVisible: false,
                        autoscaleInfoProvider: () => null,
                    });
                    lineSeries.setData([
                        { time: marker.date_start, value: marker.price },
                        { time: marker.date_end, value: marker.price }
                    ]);

                    // Anchor text to the origin swing point (date_start) to avoid massive breakout candle bodies
                    let textTime = marker.date_start;
                    
                    smcMarkersDataRef.current.push({
                        time: textTime,
                        price: marker.price,
                        text: isBOS ? 'BOS' : 'CHOCH',
                        color: tvColor,
                        direction: marker.direction, // "bullish" or "bearish"
                        id: `smc-${isBOS ? 'bos' : 'choch'}-${index}-${marker.date_start}`
                    });
                };

                recentBOS.forEach((marker: any, i: number) => renderFiniteLine(marker, true, i));
                recentCHoCH.forEach((marker: any, i: number) => renderFiniteLine(marker, false, i));

                // Add position updater for HTML overlays
                const updateLabelPositions = () => {
                    if (!chart || !candlestickSeries) return;
                    
                    const newLabels = smcMarkersDataRef.current.map(marker => {
                        const x = chart.timeScale().timeToCoordinate(marker.time as any);
                        const y = candlestickSeries.priceToCoordinate(marker.price);
                        return {
                            ...marker,
                            x: x !== null ? x : -1000,
                            y: y !== null ? y : -1000
                        };
                    }).filter(l => l.x !== -1000 && l.y !== -1000); // Filter out off-screen if necessary
                    
                    setSmcLabels(newLabels);
                };

                updateLabelPositionsRef.current = updateLabelPositions;
                chart.timeScale().subscribeVisibleTimeRangeChange(updateLabelPositions);
                chart.timeScale().subscribeVisibleLogicalRangeChange(updateLabelPositions);

                // Initial positioning calculation
                setTimeout(updateLabelPositions, 50);

                // Sort all markers chronologically and apply to candlestick series
                markers.sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
                candlestickSeries.setMarkers(markers);

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
                if (sig.type !== 'BUY') continue; // Only process AI BUY execution clicks

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
                setTooltip({ visible: false, x: 0, y: 0, data: null });
                return;
            }

            const data = dataRef.current;
            if (!data) return;

            const timeStr = param.time as string;
            const barData: any = param.seriesData.get(seriesRef.current!);

            if (!barData) {
                setTooltip({ visible: false, x: 0, y: 0, data: null });
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
            if (hoveredSig) {
                if (hoveredSig.type === 'BUY') {
                    labels.push(`AI BUY (${hoveredSig.score.toFixed(2)})`);
                } else if (hoveredSig.type === 'SELL') {
                    labels.push(`AI SELL (${hoveredSig.reason})`);
                }
            }

            // Smart Positioning Logic (Boundary Detection)
            const chartWidth = chartContainerRef.current?.clientWidth || 800;
            const chartHeight = chartContainerRef.current?.clientHeight || 500;

            const TOOLTIP_WIDTH = 180;
            const TOOLTIP_HEIGHT = 160;
            const OFFSET = 15;

            let finalX = param.point.x + OFFSET;

            // Always render tooltip ABOVE cursor when in bottom half of chart
            const inBottomHalf = param.point.y > chartHeight * 0.5;
            let finalY = inBottomHalf
                ? param.point.y - TOOLTIP_HEIGHT - OFFSET
                : param.point.y + OFFSET;

            // Flip if hitting right edge
            if (finalX + TOOLTIP_WIDTH > chartWidth) {
                finalX = param.point.x - TOOLTIP_WIDTH - OFFSET;
            }

            // Ensure tooltip doesn't go off the top/left edges
            finalX = Math.max(0, finalX);
            finalY = Math.max(0, finalY);

            setTooltip({
                visible: true,
                x: finalX,
                y: finalY,
                data: {
                    date: timeStr,
                    open: barData.open.toLocaleString(),
                    high: barData.high.toLocaleString(),
                    low: barData.low.toLocaleString(),
                    close: barData.close.toLocaleString(),
                    volume: data.ohlcv.find((c: any) => c.time === timeStr)?.volume?.toLocaleString() || "N/A",
                    signal: labels.length > 0 ? labels.join(" | ") : "None"
                }
            });
        };

        chart.subscribeCrosshairMove(handleCrosshairMove);

        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight
                });
            }
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            chart.unsubscribeClick(handleChartClick);
            chart.unsubscribeCrosshairMove(handleCrosshairMove);
            if (updateLabelPositionsRef.current) {
                chart.timeScale().unsubscribeVisibleTimeRangeChange(updateLabelPositionsRef.current);
                chart.timeScale().unsubscribeVisibleLogicalRangeChange(updateLabelPositionsRef.current);
            }
            chart.remove();
        };
    }, [ticker]);

    const handleTimeframeClick = (tf: string) => {
        if (!chartRef.current || !dataRef.current?.ohlcv?.length) return;
        setTimeframe(tf);

        const ohlcv = dataRef.current.ohlcv;
        const lastIndex = ohlcv.length - 1;

        if (tf === 'ALL') {
            chartRef.current.timeScale().fitContent();
            return;
        }

        // Approximate trading days per period (no date math needed)
        let startIndex = 0;
        if (tf === '7D') startIndex = Math.max(0, lastIndex - 5);
        if (tf === '1M') startIndex = Math.max(0, lastIndex - 22);
        if (tf === '3M') startIndex = Math.max(0, lastIndex - 65);
        if (tf === '1Y') startIndex = Math.max(0, lastIndex - 252);

        // Use EXACT time values from the data array - zero format conversion risk
        chartRef.current.timeScale().setVisibleRange({
            from: ohlcv[startIndex].time as any,
            to: ohlcv[lastIndex].time as any,
        });
    };

    return (
        <div className="flex-1 w-full min-h-0 relative rounded-lg border border-gray-800 overflow-hidden bg-gray-900 flex flex-col" style={{ userSelect: "none" }}>
            <div className="px-3 py-2 border-b border-gray-800 z-10 bg-gray-900 h-auto">
                <div className="flex flex-row flex-nowrap items-center justify-between w-full h-auto gap-2 p-1">
                    <div className="flex flex-nowrap items-center gap-x-3 flex-shrink-0">
                        <h2 className="text-white font-semibold whitespace-nowrap text-sm hidden md:block">Green Dragon Live Feed</h2>
                        <select
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value)}
                            className="bg-gray-800 text-white text-sm px-3 py-1 rounded border border-gray-700 outline-none focus:border-green-500 cursor-pointer flex-shrink-0"
                        >
                            {["ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"].map(t => (
                                <option key={t} value={t}>{t}</option>
                            ))}
                        </select>

                        {/* Timeframe Buttons */}
                        <div className="flex space-x-1 border border-gray-700 rounded overflow-hidden flex-shrink-0">
                            {[
                                { label: '7D', key: '7D' },
                                { label: '1M', key: '1M' },
                                { label: '3M', key: '3M' },
                                { label: '1Y', key: '1Y' },
                                { label: 'ALL', key: 'ALL' },
                            ].map(tf => (
                                <button
                                    key={tf.key}
                                    onClick={() => handleTimeframeClick(tf.key)}
                                    className={`px-2 py-1 text-[11px] font-semibold transition-colors whitespace-nowrap ${timeframe === tf.key ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
                                >
                                    {tf.label}
                                </button>
                            ))}
                        </div>

                        {/* Dynamic Date Range Indicator */}
                        {dateRange.start && dateRange.end && (
                            <div className="flex items-center space-x-1.5 text-[11px] text-gray-400 bg-gray-800/50 px-2 py-1 rounded border border-gray-700 flex-shrink-0 whitespace-nowrap hidden sm:flex">
                                <span>Period:</span>
                                <span className="text-gray-200 font-medium">{dateRange.start}</span>
                                <span>-</span>
                                <span className="text-gray-200 font-medium">{dateRange.end}</span>
                            </div>
                        )}

                        <span className="text-[11px] font-bold text-blue-400 bg-blue-900/30 px-1.5 py-1 rounded border border-blue-800 flex-shrink-0 whitespace-nowrap hidden xl:inline">Finite SMC</span>
                        <span className="text-[11px] font-bold text-green-400 bg-green-900/30 px-1.5 py-1 rounded border border-green-800 flex-shrink-0 whitespace-nowrap hidden xl:inline">LSTM 0.635</span>
                    </div>

                    <div className="flex flex-nowrap items-center gap-2 flex-shrink-0 ml-auto">
                        {loading && <span className="text-[11px] text-green-400 animate-pulse whitespace-nowrap">Syncing Engine...</span>}
                        <span className="text-[11px] text-gray-500 whitespace-nowrap hidden 2xl:inline">Real-time Optuna Active</span>
                    </div>
                </div>
            </div>
            <div className="flex-1 w-full min-h-0 relative overflow-hidden bg-[#131722]">
                <div ref={chartContainerRef} className="absolute top-0 left-0 right-0 bottom-0" />

                {/* HTML Overlays for SMC Labels */}
                {smcLabels.map((label) => {
                    const yOffset = label.direction === 'bullish' ? '-100%' : '0%'; // above line for bullish, below for bearish
                    
                    return (
                        <div
                            key={label.id}
                            className="absolute z-40 text-[10px] font-bold pointer-events-none"
                            style={{
                                left: label.x,
                                top: label.y,
                                color: label.color,
                                transform: `translate(0px, ${yOffset})`, // shift horizontally 0 so it aligns with the start, offset vertically
                            }}
                        >
                            {label.text}
                        </div>
                    );
                })}

                {/* Hover Crosshair Legend Tooltip */}
                {tooltip.visible && tooltip.data && (
                    <div
                        className="absolute z-50 flex flex-col gap-1 p-3 bg-gray-900/95 border border-gray-700 rounded-md shadow-2xl pointer-events-none text-xs"
                        style={{ left: tooltip.x + 15, top: tooltip.y + 15 }}
                    >
                        <div className="font-bold text-gray-100 mb-1">{tooltip.data.date}</div>
                        <div className="flex flex-col gap-0.5 text-gray-300 font-mono">
                            <div><span className="text-gray-500 w-8 inline-block">O:</span> {tooltip.data.open}</div>
                            <div><span className="text-gray-500 w-8 inline-block">H:</span> {tooltip.data.high}</div>
                            <div><span className="text-gray-500 w-8 inline-block">L:</span> {tooltip.data.low}</div>
                            <div><span className="text-gray-500 w-8 inline-block">C:</span> {tooltip.data.close}</div>
                            <div><span className="text-gray-500 w-8 inline-block">Vol:</span> {tooltip.data.volume}</div>
                        </div>
                        {tooltip.data.signal !== "None" && (
                            <div className={`mt-1 font-bold ${tooltip.data.signal.includes('BUY') || tooltip.data.signal.includes('BOS') ? 'text-green-400' : 'text-red-400'}`}>
                                {tooltip.data.signal}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
