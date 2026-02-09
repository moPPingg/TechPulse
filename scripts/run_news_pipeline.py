#!/usr/bin/env python3
"""
Run the Vietnamese stock news pipeline: crawl → clean → sentiment → align → store.
Usage:
  python scripts/run_news_pipeline.py
  python scripts/run_news_pipeline.py --steps crawl clean
  python scripts/run_news_pipeline.py --config configs/news.yaml
  python scripts/run_news_pipeline.py --schedule 2   # Chạy mỗi 2 giờ liên tục
  python scripts/run_news_pipeline.py --schedule 4   # Chạy mỗi 4 giờ
"""

import sys
import time
import argparse
import logging
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="News pipeline: crawl, clean, sentiment, align")
    ap.add_argument("--config", default="configs/news.yaml", help="Path to news config YAML")
    ap.add_argument("--steps", nargs="+", default=None, help="Steps: crawl clean sentiment align enrich (default: all)")
    ap.add_argument("--schedule", type=float, default=0, metavar="HOURS", help="Chạy lặp mỗi N giờ (vd: 2 hoặc 4). Ctrl+C để dừng.")
    args = ap.parse_args()

    from src.news.pipeline import run_pipeline

    if args.schedule and args.schedule > 0:
        interval_sec = args.schedule * 3600
        logger.info("Chế độ lên lịch: chạy pipeline mỗi %.1f giờ", args.schedule)
        while True:
            try:
                results = run_pipeline(config_path=args.config, steps=args.steps)
                logger.info("Pipeline xong: %s. Chờ %.1f giờ...", results, args.schedule)
            except KeyboardInterrupt:
                logger.info("Dừng bởi người dùng.")
                break
            except Exception as e:
                logger.exception("Lỗi pipeline: %s", e)
            time.sleep(interval_sec)
        return 0

    results = run_pipeline(config_path=args.config, steps=args.steps)
    logger.info("Pipeline finished: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
