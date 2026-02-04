#!/usr/bin/env python3
"""
Convert foundation markdown docs to standalone HTML for easier reading.
Usage: python scripts/md_to_html.py
"""
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Installing markdown... run: pip install markdown")
    raise

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "01_foundations"
FILES = [
    "02_DEEP_LEARNING_BASICS.md",
    "03_TIME_SERIES_FUNDAMENTALS.md",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #fafafa;
      --text: #1a1a1a;
      --muted: #555;
      --code-bg: #f0f0f0;
      --border: #e0e0e0;
      --accent: #2563eb;
      --blockquote: #e8eef7;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #1a1a1a;
        --text: #e4e4e4;
        --muted: #a0a0a0;
        --code-bg: #2d2d2d;
        --border: #333;
        --accent: #60a5fa;
        --blockquote: #1e293b;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", "Source Sans 3", system-ui, sans-serif;
      line-height: 1.65;
      color: var(--text);
      background: var(--bg);
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
    }}
    h1 {{ font-size: 1.85rem; margin-top: 0; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }}
    h2 {{ font-size: 1.4rem; margin-top: 2rem; color: var(--accent); }}
    h3 {{ font-size: 1.15rem; margin-top: 1.5rem; }}
    h1, h2, h3 {{ scroll-margin-top: 1rem; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    pre, code {{
      font-family: "Consolas", "Fira Code", monospace;
      background: var(--code-bg);
      border-radius: 6px;
    }}
    code {{ padding: 0.2em 0.4em; font-size: 0.9em; }}
    pre {{ padding: 1rem; overflow-x: auto; margin: 1rem 0; }}
    pre code {{ padding: 0; background: none; }}
    blockquote {{
      border-left: 4px solid var(--accent);
      margin: 1rem 0;
      padding: 0.5rem 1rem;
      background: var(--blockquote);
      color: var(--muted);
    }}
    hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}
    ul, ol {{ padding-left: 1.5rem; }}
    li {{ margin: 0.35rem 0; }}
    .back-link {{ display: inline-block; margin-bottom: 1rem; color: var(--muted); font-size: 0.9rem; }}
  </style>
</head>
<body>
  <a class="back-link" href=".">← Trở lại thư mục</a>
  <main>
{body}
  </main>
</body>
</html>
"""


def get_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return fallback


def md_to_html(md_path: Path, out_path: Path, fallback_title: str) -> None:
    content = md_path.read_text(encoding="utf-8")
    title = get_title(content, fallback_title)
    # "extra": tables, fenced code...; "toc": thêm id cho heading để link mục lục trong bài nhảy đúng (không đổi chữ)
    html_body = markdown.markdown(
        content,
        extensions=["extra", "toc"],
        extension_configs={"toc": {"title": "Mục lục"}},
    )
    html = HTML_TEMPLATE.format(title=title or fallback_title, body=html_body)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    for name in FILES:
        md_path = DOCS_DIR / name
        if not md_path.exists():
            print(f"Skip (not found): {md_path}")
            continue
        out_path = md_path.with_suffix(".html")
        fallback_title = name.replace(".md", "").replace("_", " ")
        md_to_html(md_path, out_path, fallback_title)


if __name__ == "__main__":
    main()
