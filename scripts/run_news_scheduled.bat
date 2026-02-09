@echo off
cd /d "%~dp0.."
python scripts/run_news_pipeline.py
exit /b 0
