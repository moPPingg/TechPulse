"""
Standalone learning example: how the news composite score is computed.

This file is for education only. It is not imported by the application.
It demonstrates the same weighted formula used in src/news/engine.py:
  weight = relevance * horizon_weight; composite = sum(sentiment * weight) / sum(weight).
"""

articles = [
    {"relevance" : 0.8,"sentiment" : 0.6, "horizon_weight" : 1.0},
    {"relevance" : 0.5,"sentiment" : -0.4 ,"horizon_weight" : 0.7},
    {"relevance" : 0.9,"sentiment" : 0.2, "horizon_weight" : 0.5}
]

weighted_sum = 0
weight_sum = 0 

for a in articles:
    w = a['relevance'] * a["horizon_weight"]
    weighted_sum += a['sentiment']* w
    weight_sum += w

composite = weighted_sum / weight_sum

if composite >0.15:
    label = "bullish"
elif composite < 0.15:
    label = "bearish"
else:
    label = "neutral"

print("Composite:", composite)
print("Label:", label)