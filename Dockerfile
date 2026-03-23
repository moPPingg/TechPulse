FROM python:3.11-slim

WORKDIR /app

# Install all deps needed by both fetcher and alert system
COPY requirements-fetcher.txt ./
RUN pip install --no-cache-dir -r requirements-fetcher.txt

# Copy scripts (CMD is overridden per service in docker-compose.yml)
COPY scripts/yfinance_fetcher.py ./scripts/
COPY scripts/alert_system.py     ./scripts/

# Default env vars (overridden in docker-compose.yml)
ENV DATA_DIR=/app/data/raw
ENV LOG_LEVEL=INFO

CMD ["python", "scripts/yfinance_fetcher.py"]
