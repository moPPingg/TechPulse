#!/usr/bin/env python3
"""
Run the Vietnamese stock news pipeline: crawl → clean → sentiment → align → store.
Usage:
  python scripts/run_news_pipeline.py
  python scripts/run_news_pipeline.py --steps crawl clean
  python scripts/run_news_pipeline.py --config configs/news.yaml
"""

import sys
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
    ap.add_argument("--steps", nargs="+", default=None, help="Steps to run: crawl clean sentiment align (default: all)")
    args = ap.parse_args()

    from src.news.pipeline import run_pipeline

    results = run_pipeline(config_path=args.config, steps=args.steps)
    logger.info("Pipeline finished: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
