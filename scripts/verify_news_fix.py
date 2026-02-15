import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app_services.news_service import get_articles, Article

def verify():
    print("Fetching articles for 'FPT'...")
    try:
        articles = get_articles("FPT", limit=5)
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return

    print(f"Found {len(articles)} articles.")
    for a in articles:
        print(f"---")
        print(f"Title: {a.title}")
        print(f"Source: {a.source}")
        print(f"Original URL (raw from DB logic might be hidden here, checking result): {a.url}")
        
        if not a.url.startswith("http"):
            print("❌ ERROR: URL is not absolute!")
        else:
            print("✅ URL is absolute.")

if __name__ == "__main__":
    verify()
