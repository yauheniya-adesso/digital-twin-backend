# -*- coding: utf-8 -*-
# Download Medium articles as Markdown using public URLs

import os
import requests
from bs4 import BeautifulSoup
import html2text

# === CONFIG ===
USERNAME = "yauheniya.ai"  # your Medium username in the URL
OUTPUT_DIR = "data_raw/medium_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Helper function to convert HTML to Markdown ===
def html_to_md(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    return h.handle(html_content)

# === Step 1: Get list of article URLs ===
# Medium user RSS feed provides all public posts
rss_url = f"https://medium.com/feed/@{USERNAME}"
resp = requests.get(rss_url)
if resp.status_code != 200:
    print(f"❌ Failed to fetch RSS feed: {resp.status_code}")
    exit(1)

soup = BeautifulSoup(resp.text, "xml")
items = soup.find_all("item")
print(f"Found {len(items)} articles in RSS feed\n")

# === Step 2: Download each article ===
for item in items:
    title = item.title.text
    link = item.link.text
    print(f"📦 Downloading: {title}")

    # Fetch article HTML
    r = requests.get(link)
    if r.status_code != 200:
        print(f"   ⚠️ Failed to fetch article: {r.status_code}")
        continue

    # Convert HTML to Markdown
    md = html_to_md(r.text)

    # Sanitize filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
    filename = f"{safe_title}.md"
    path = os.path.join(OUTPUT_DIR, filename)

    # Save Markdown file
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"   ✓ Saved: {path}")

print("\n✅ All Medium articles downloaded to:", OUTPUT_DIR)
