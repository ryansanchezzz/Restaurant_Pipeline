import os
import csv
import json
import asyncio
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import aiohttp

from utils.is_img_or_pdf import is_img_or_pdf
from scraper.extract_data_ocr import extract_data_ocr
from scraper.scrape_ai_tool import scrape_ai_tool

OUTPUT_JSON = "happy_hour_dataset.json"
CSV_FILE = "Happy Hours in SF - Sheet2.csv"

async def fetch_html(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Failed to fetch {url}: Status {response.status}")
                return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def extract_links_from_html(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a['href'].strip()
        if href.startswith("http"):
            links.add(href)
        elif href.startswith("/"):
            parsed_base = urlparse(base_url)
            abs_link = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
            links.add(abs_link)
    return links

async def scrape_and_label_link(session, link, final_url):
    try:
        if is_img_or_pdf(link):
            print(f"{link} is PDF/image. Using OCR.")
            text_content = extract_data_ocr(link)
        else:
            print(f"Scraping {link}")
            text_content = await scrape_ai_tool(link)

        # Exact match only
        label = 1 if link.rstrip("/") == final_url.rstrip("/") else 0

        return {
            "url": link,
            "label": label,
            "text": text_content
        }

    except Exception as e:
        print(f"Error processing {link}: {e}")
        return None

async def scrape_all_links_for_base(base_url, final_url, base_dataset):
    print(f"\nProcessing base URL: {base_url}")

    all_links = set()
    

    async with aiohttp.ClientSession() as session:
        html = await fetch_html(session, base_url)
        if html:
            links = extract_links_from_html(base_url, html)
            all_links.update(links)

        entries = []
        for link in all_links:
            entry = await scrape_and_label_link(session, link, final_url)
            if entry:
                entries.append(entry)

        base_dataset[base_url] = entries

async def scrape_urls_from_csv(csv_file):
    base_dataset = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_url = row["Restaurant Base URL"].strip()
            
            await scrape_all_links_for_base(base_url, final_url, base_dataset)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(base_dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to {OUTPUT_JSON}. Total base URLs: {len(base_dataset)}")

# === RUN IT ===
if __name__ == "__main__":
    asyncio.run(scrape_urls_from_csv(CSV_FILE))
