# -*- coding: utf-8 -*-
# Download README.md files from all your GitHub repositories
# Secure version using environment variables (.env)

import os
import requests
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OUTPUT_DIR = "data_raw/github_work_readmes"

if not GITHUB_USERNAME or not GITHUB_TOKEN:
    print("❌ Missing GITHUB_USERNAME or GITHUB_TOKEN in .env")
    exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Fetch list of repositories ===
repos_url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos?per_page=100"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

print("🔍 Fetching repository list…")
repos = requests.get(repos_url, headers=headers).json()

if isinstance(repos, dict) and "message" in repos:
    print("❌ Error:", repos["message"])
    exit(1)

print(f"✓ Found {len(repos)} repositories\n")

# === Download README.md from each repo ===
for repo in repos:
    repo_name = repo["name"]
    readme_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme"

    print(f"📦 {repo_name} → fetching README…")

    r = requests.get(readme_url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        download_url = data.get("download_url")

        if not download_url:
            print(f"   ⚠️ README has no download URL.")
            continue

        readme = requests.get(download_url, headers=headers).text
        out_path = os.path.join(OUTPUT_DIR, f"{repo_name}_README.md")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(readme)

        print(f"   ✓ Saved: {out_path}")
    else:
        print(f"   ⚠️ No README found or access denied (status {r.status_code})")

print("\n✅ All README files downloaded to:", OUTPUT_DIR)
