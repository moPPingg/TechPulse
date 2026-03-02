#!/usr/bin/env python3
"""
Wrapper script to run data cleaning and feature engineering (both price & news).
Usage:
  python scripts/run_feature_engineering.py
"""
import sys
import logging
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.pipeline.vnindex30.fetch_vn30 import load_pipeline_config
from src.clean.clean_price import clean_many
from src.features.build_features import build_features
from src.news.pipeline import run_pipeline as run_news_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("🚀 STARTING FEATURE ENGINEERING (PRICE & NEWS)")
    logger.info("=" * 60)
    
    cfg = load_pipeline_config()
    
    # 1. Clean Price Data
    logger.info("\n--- 1. Cleaning Price Data ---")
    try:
        clean_many(
            raw_dir=cfg['raw_dir'],
            clean_dir=cfg['clean_dir'],
            skip_on_error=cfg['skip_on_error'],
            remove_duplicates=cfg['remove_duplicates'],
            remove_nulls=cfg['remove_nulls'],
            validate=cfg['validate'],
        )
    except Exception as e:
        logger.error(f"Price cleaning failed: {e}")

    # 2. Build Price Features
    logger.info("\n--- 2. Building Price Features ---")
    try:
        build_features(
            clean_dir=cfg['clean_dir'],
            features_dir=cfg['features_dir'],
            skip_on_error=cfg['skip_on_error'],
            drop_na=True,
        )
    except Exception as e:
        logger.error(f"Price feature building failed: {e}")
        
    # 3. Process News Features (Clean -> Sentiment -> Align -> Enrich)
    logger.info("\n--- 3. Processing News Pipeline ---")
    try:
        run_news_pipeline(steps=["clean", "sentiment", "align", "enrich"])
    except Exception as e:
        logger.error(f"News pipeline processing failed: {e}")
        
    logger.info("=" * 60)
    logger.info("🎉 FEATURE ENGINEERING FINISHED")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
