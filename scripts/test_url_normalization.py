import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app_services.news_service import _normalize_url

def test_normalization():
    test_cases = [
        # (url, source, expected_start)
        ("/2024/02/09/news.chn", None, "https://cafef.vn/2024/02/09/news.chn"),
        ("/2024/02/09/news.chn", "", "https://cafef.vn/2024/02/09/news.chn"),
        ("/2024/02/09/news.html", None, "https://vnexpress.net/2024/02/09/news.html"),
        ("/2024/02/09/news.htm", None, "https://cafef.vn/2024/02/09/news.htm"), # Fallback to CafeF for .htm if not caught? 
        # Wait, my logic for .htm was specific. Let's see what I implemented.
        # implemented: .htm or /20 -> pass (no base set in that specific elif block), then fallback to CafeF if u.startswith("/")
        
        ("https://othersite.com/news", "Other", "https://othersite.com/news"),
        ("/relative/path", "CafeF", "https://cafef.vn/relative/path"),
        ("/relative/path", "cafef", "https://cafef.vn/relative/path"),
    ]

    print("Testing _normalize_url logic...")
    failures = 0
    for url, source, expected in test_cases:
        result = _normalize_url(url, source)
        print(f"Input: url='{url}', source='{source}'")
        print(f"Output: {result}")
        
        # Check if result matches expected (or close enough for fallback)
        if result == expected:
            print("✅ PASS")
        else:
            print(f"❌ FAIL. Expected {expected}")
            failures += 1
    
    if failures == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failures} tests failed.")

if __name__ == "__main__":
    test_normalization()
