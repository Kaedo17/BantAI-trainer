"""
Google Images Scraper - Easy Data Gathering Tool
Usage:
# Basic run
#python bantai_scrape.py "overloaded sockets" -n 200 --browser --visible

# Only keep images at least 200x200
#python bantai_scrape.py "overloaded sockets" -n 200 --browser --visible --min-res 200x200

# Stack multiple queries into the same folder for more images
#python bantai_scrape.py "overloaded power outlets" -n 200 -o ./scraped_images/overloaded --browser --visible
#python bantai_scrape.py "too many plugs in socket" -n 200 -o ./scraped_images/overloaded --browser --visible
"""

import os
import re
import sys
import time
import io
import json
import hashlib
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: Pillow for robust image validation
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Optional: selenium for scrolling (loads more results beyond ~100)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}

# Domains to skip (Google's own assets, not actual image results)
SKIP_DOMAINS = {
    "google.com", "gstatic.com", "googleapis.com", "google.co",
    "youtube.com", "ytimg.com", "schema.org", "w3.org",
    "googleusercontent.com", "ggpht.com", "google-analytics.com",
}


def is_valid_image_url(url):
    """Check if a URL looks like a real image result (not a Google asset)."""
    if not url or not url.startswith("http"):
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        # Skip Google's own domains
        for skip in SKIP_DOMAINS:
            if domain.endswith(skip):
                return False
        # Skip non-image resources
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in [".js", ".css", ".html", ".htm", ".xml"]):
            return False
        return True
    except Exception:
        return False


def build_google_url(query, start=0):
    """Build a Google Images search URL."""
    params = urllib.parse.urlencode({
        "q": query,
        "tbm": "isch",
        "ijn": str(start // 100),
        "start": str(start),
    })
    return f"https://www.google.com/search?{params}"


def fetch_page(url):
    """Fetch a page and return its HTML content."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_image_urls_from_html(html):
    """Extract full-resolution image URLs from Google Images HTML using multiple strategies."""
    urls = []
    seen = set()

    def add_url(u):
        if u not in seen and is_valid_image_url(u):
            seen.add(u)
            urls.append(u)

    # Strategy 1: Find URLs in JSON-like structures ["url",width,height]
    for m in re.finditer(r'\["(https?://[^"]{20,})"(?:,\s*(\d+)\s*,\s*(\d+))?', html):
        candidate = m.group(1)
        # Unescape unicode sequences like \u003d -> =
        try:
            candidate = candidate.encode().decode("unicode_escape")
        except Exception:
            pass
        add_url(candidate)

    # Strategy 2: Extract from data attributes and src attributes
    for m in re.finditer(r'(?:src|data-src|data-iurl|data-ou|ou)="(https?://[^"]+)"', html, re.IGNORECASE):
        add_url(m.group(1))

    # Strategy 3: Look for image URLs in script/AF_initDataCallback blocks
    for m in re.finditer(r'(https?://[^\s"\'\\,\]\[}{<>]{20,}\.(?:jpg|jpeg|png|gif|webp|bmp)(?:\?[^\s"\'\\,\]\[}{<>]*)?)', html, re.IGNORECASE):
        candidate = m.group(1)
        try:
            candidate = candidate.encode().decode("unicode_escape")
        except Exception:
            pass
        add_url(candidate)

    return urls


def scrape_with_requests(query, count):
    """Scrape image URLs using only urllib (no browser needed). Works for ~100-200 images."""
    print(f"[*] Scraping with requests mode (no browser)...")
    all_urls = []
    max_pages = (count // 100) + 2

    for page_num in range(max_pages):
        if len(all_urls) >= count:
            break
        url = build_google_url(query, start=page_num * 100)
        print(f"    Fetching page {page_num + 1}...")
        try:
            html = fetch_page(url)
            urls = extract_image_urls_from_html(html)
            new_count = 0
            for u in urls:
                if u not in all_urls:
                    all_urls.append(u)
                    new_count += 1
            print(f"    Found {new_count} new URLs (total: {len(all_urls)})")
            if new_count == 0:
                break
            time.sleep(1)
        except Exception as e:
            print(f"    [!] Error fetching page: {e}")
            break

    return all_urls[:count]


def _create_driver(visible=False):
    """Create a Chrome driver (headless by default, visible with --visible)."""
    options = Options()
    if not visible:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    driver = webdriver.Chrome(options=options)
    # Hide webdriver flag from detection
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def _scroll_to_load(driver, count):
    """Scroll page to load enough thumbnails."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    stale_scrolls = 0
    max_stale = 4

    for i in range(max((count // 40) + 5, 10)):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

        # Click "Show more results" button if it appears
        try:
            for selector in ["input[type='button']", ".mye4qd", "[jsaction*='show_more']"]:
                btns = driver.find_elements(By.CSS_SELECTOR, selector)
                for btn in btns:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(2)
        except Exception:
            pass

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stale_scrolls += 1
            if stale_scrolls >= max_stale:
                break
        else:
            stale_scrolls = 0
        last_height = new_height
        # Count how many thumbnails are loaded
        thumb_count = driver.execute_script(
            "return document.querySelectorAll('div[data-id] img, div[jsaction] img.YQ4gaf, img.rg_i').length"
        )
        print(f"    Scrolling... ({thumb_count} thumbnails loaded)")
        if thumb_count >= count * 1.2:
            break


def scrape_with_selenium(query, count, visible=False):
    """Scrape by clicking each thumbnail and extracting the full-res image URL."""
    print(f"[*] Scraping with Selenium mode ({'visible' if visible else 'headless'} browser)...")
    driver = _create_driver(visible=visible)

    try:
        url = build_google_url(query)
        driver.get(url)
        time.sleep(3)

        # Accept cookies/consent if prompted — try multiple approaches
        try:
            # Google consent buttons
            for text in ["Accept all", "Accept", "I agree", "Agree", "Reject all", "Consent"]:
                btns = driver.find_elements(By.XPATH, f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]")
                for btn in btns:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(1)
                        break
            # Also try form-based consent
            for sel in ["form[action*='consent'] button", "#L2AGLb", "#W0wltc"]:
                btns = driver.find_elements(By.CSS_SELECTOR, sel)
                for btn in btns:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(1)
        except Exception:
            pass

        time.sleep(2)

        # Check if we actually landed on an images page
        current = driver.current_url
        page_title = driver.title
        print(f"    Page title: {page_title}")
        print(f"    URL: {current}")

        # If redirected to consent page, save debug info
        if "consent" in current.lower() or "sorry" in current.lower():
            print("[!] Google redirected to a consent/block page.")
            if visible:
                print("    >>> Handle the consent/CAPTCHA in the browser window, then press ENTER here <<<")
                input()
                time.sleep(2)
                # Re-navigate to images after consent
                driver.get(build_google_url(query))
                time.sleep(3)
            else:
                debug_path = os.path.join(".", "debug_screenshot.png")
                driver.save_screenshot(debug_path)
                print(f"    Debug screenshot saved: {os.path.abspath(debug_path)}")
                print("    TIP: Rerun with --visible to handle this manually.")

        # Phase 1: Scroll to load enough thumbnails
        print("[*] Phase 1: Loading thumbnails...")
        _scroll_to_load(driver, count)

        # Quick check: if 0 thumbnails, dump debug info
        thumb_check = driver.execute_script(
            "return document.querySelectorAll('div[data-id] img, div[jsaction] img.YQ4gaf, img.rg_i, #islrg img').length"
        )
        if thumb_check == 0:
            print("[!] No thumbnails found on page. Saving debug info...")
            os.makedirs(".", exist_ok=True)
            debug_path = os.path.join(".", "debug_screenshot.png")
            driver.save_screenshot(debug_path)
            print(f"    Screenshot: {os.path.abspath(debug_path)}")
            debug_html = os.path.join(".", "debug_page.html")
            with open(debug_html, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"    HTML saved: {os.path.abspath(debug_html)}")
            if visible:
                print("    >>> Fix any issues in the browser window, then press ENTER here <<<")
                input()
                # Re-navigate and try again
                driver.get(build_google_url(query))
                time.sleep(3)
                _scroll_to_load(driver, count)
            else:
                print("    TIP: Run with --visible to see the browser and bypass CAPTCHA manually.")

        # Phase 2: Extract image URLs using JavaScript (most reliable method)
        print("[*] Phase 2: Extracting image URLs via JavaScript...")
        collected_urls = []
        seen = set()

        # JS approach: click each thumbnail, then grab the full-res src from the
        # largest visible <img> that isn't a Google-owned asset.
        # First, gather all thumbnail img elements via JS.
        thumb_elements = driver.find_elements(By.CSS_SELECTOR, "img")
        # Filter to actual search result thumbnails (skip tiny icons, logos, etc.)
        result_thumbs = []
        for img in thumb_elements:
            try:
                w = img.get_attribute("width") or "0"
                h = img.get_attribute("height") or "0"
                src = img.get_attribute("src") or ""
                parent_tag = img.find_element(By.XPATH, "..").tag_name
                # Thumbnails are typically inside <a> or <div> and have reasonable size
                if parent_tag in ("a", "div") and (int(w) >= 50 or int(h) >= 50 or "encrypted" in src or "data:image" in src):
                    result_thumbs.append(img)
            except Exception:
                continue

        if not result_thumbs:
            # Broader fallback: any img with class names Google uses for thumbnails
            result_thumbs = driver.find_elements(By.CSS_SELECTOR,
                "img.YQ4gaf, img.rg_i, img.Q4LuWd, img[data-src], div[data-id] img"
            )

        print(f"    Found {len(result_thumbs)} thumbnail images")

        for i, thumb in enumerate(result_thumbs):
            if len(collected_urls) >= count:
                break

            try:
                # Click the thumbnail
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", thumb)
                time.sleep(0.2)
                driver.execute_script("arguments[0].click();", thumb)
                time.sleep(1.0)

                # Use JS to find the full-res image:
                # 1. First try known Google preview class names (fast)
                # 2. Fall back to finding the largest non-Google img on page
                found_url = driver.execute_script("""
                    var dominated = [
                        'google.com','gstatic.com','googleapis.com','google.co',
                        'youtube.com','ytimg.com','googleusercontent.com','ggpht.com'
                    ];
                    function isDominated(src) {
                        for (var d = 0; d < dominated.length; d++) {
                            if (src.indexOf(dominated[d]) !== -1) return true;
                        }
                        return false;
                    }
                    function isGood(src) {
                        return src && src.startsWith('http') && src.length > 30
                            && !isDominated(src) && src.indexOf('encrypted') === -1;
                    }

                    // Method 1: Known Google full-res preview class names
                    var knownClasses = ['n3VNCb','iPVvYb','r48jcc','pT0Scc','H8Rx8c','sFlh5c','FyHeAf'];
                    for (var c = 0; c < knownClasses.length; c++) {
                        var els = document.getElementsByClassName(knownClasses[c]);
                        for (var j = 0; j < els.length; j++) {
                            var s = els[j].src || els[j].getAttribute('data-src') || '';
                            if (isGood(s)) return s;
                        }
                    }

                    // Method 2: Find the largest visible non-Google image
                    var best = null;
                    var bestArea = 0;
                    var imgs = document.querySelectorAll('img[src^="http"]');
                    for (var i = 0; i < imgs.length; i++) {
                        var src = imgs[i].src;
                        if (!isGood(src)) continue;
                        var w = imgs[i].naturalWidth || imgs[i].width || 0;
                        var h = imgs[i].naturalHeight || imgs[i].height || 0;
                        var area = w * h;
                        if (area > bestArea) {
                            bestArea = area;
                            best = src;
                        }
                    }
                    return best;
                """)

                if found_url and found_url not in seen and is_valid_image_url(found_url):
                    seen.add(found_url)
                    collected_urls.append(found_url)
                    if len(collected_urls) % 10 == 0 or len(collected_urls) <= 5:
                        print(f"    [{len(collected_urls)}/{count}] ...")

            except Exception:
                continue

        print(f"    Got {len(collected_urls)} URLs from clicking")

        # Phase 3: Supplement with HTML/JS source parsing if we still need more
        if len(collected_urls) < count:
            print(f"    Parsing page source for additional URLs...")
            html = driver.page_source
            parsed_urls = extract_image_urls_from_html(html)
            for u in parsed_urls:
                if len(collected_urls) >= count:
                    break
                if u not in seen:
                    seen.add(u)
                    collected_urls.append(u)
            print(f"    Total after HTML parse: {len(collected_urls)}")

        print(f"    Collected {len(collected_urls)} image URLs")
        return collected_urls[:count]
    finally:
        driver.quit()


# Valid image file signatures (magic bytes) — fallback when PIL is not installed
IMAGE_MAGIC = {
    b'\xff\xd8\xff': ".jpg",
    b'\x89PNG': ".png",
    b'GIF87a': ".gif",
    b'GIF89a': ".gif",
    b'RIFF': ".webp",
    b'BM': ".bmp",
}

# Format to extension mapping for PIL
PIL_FORMAT_EXT = {
    "JPEG": ".jpg", "PNG": ".png", "GIF": ".gif",
    "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff",
}


def validate_image(data, min_res=(0, 0), max_res=(0, 0)):
    """
    Validate image data and return (extension, cleaned_data) or (None, None).
    Uses PIL if available (catches corrupted/truncated files).
    Falls back to magic byte check otherwise.
    """
    if len(data) < 1000:
        return None, None

    if HAS_PIL:
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()  # detect corrupted data
            # Re-open after verify (verify closes the file)
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            fmt = img.format

            # Resolution filtering
            if min_res != (0, 0):
                if w < min_res[0] or h < min_res[1]:
                    return None, None
            if max_res != (0, 0):
                if w > max_res[0] or h > max_res[1]:
                    return None, None

            ext = PIL_FORMAT_EXT.get(fmt, ".jpg")

            # Convert RGBA/P to RGB when saving as JPEG (prevents crash)
            if ext == ".jpg" and img.mode in ("RGBA", "P", "LA"):
                rgb_img = img.convert("RGB")
                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=95)
                data = buf.getvalue()

            img.close()
            return ext, data
        except Exception:
            return None, None
    else:
        # Fallback: magic byte check only
        for magic, ext in IMAGE_MAGIC.items():
            if data[:len(magic)] == magic:
                return ext, data
        return None, None


def load_downloaded_log(output_dir):
    """Load the set of already-downloaded URLs from the log file."""
    log_path = os.path.join(output_dir, "_downloaded_urls.txt")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_to_downloaded_log(output_dir, url):
    """Append a URL to the download log file."""
    log_path = os.path.join(output_dir, "_downloaded_urls.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def get_next_index(output_dir):
    """Get the next image index based on existing files in the directory."""
    existing = [f for f in os.listdir(output_dir) if f.startswith("img_") and f[4:8].isdigit()]
    if not existing:
        return 1
    max_idx = max(int(f[4:8]) for f in existing)
    return max_idx + 1


def download_image(url, output_dir, index, total, min_res=(0, 0), max_res=(0, 0)):
    """Download a single image. Returns (success, filepath)."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()

            # Validate image (PIL if available, magic bytes fallback)
            ext, clean_data = validate_image(data, min_res, max_res)
            if ext is None:
                print(f"    [{index}/{total}] Skipped (invalid/filtered): {url[:60]}")
                return False, None

            # Use hash for unique filename
            name_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = f"img_{index:04d}_{name_hash}{ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(clean_data)

            # Log the URL so re-runs skip it
            save_to_downloaded_log(output_dir, url)

            print(f"    [{index}/{total}] Downloaded: {filename} ({len(data) // 1024}KB)")
            return True, filepath
    except Exception as e:
        print(f"    [{index}/{total}] Failed: {str(e)[:60]}")
        return False, None


def download_images(urls, output_dir, workers=8, min_res=(0, 0), max_res=(0, 0)):
    """Download images in parallel, skipping already-downloaded URLs."""
    os.makedirs(output_dir, exist_ok=True)

    # Skip URLs already downloaded in previous runs
    already_done = load_downloaded_log(output_dir)
    new_urls = [u for u in urls if u not in already_done]
    skipped = len(urls) - len(new_urls)
    if skipped > 0:
        print(f"    Skipping {skipped} already-downloaded images")

    total = len(new_urls)
    if total == 0:
        print(f"\n[*] All {len(urls)} images already downloaded!")
        return len(urls), 0

    start_idx = get_next_index(output_dir)
    print(f"\n[*] Downloading {total} new images to: {output_dir}")
    print(f"    Using {workers} parallel workers\n")

    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, url, output_dir, start_idx + i, total, min_res, max_res): url
            for i, url in enumerate(new_urls)
        }
        for future in as_completed(futures):
            ok, path = future.result()
            if ok:
                success_count += 1
            else:
                failed_count += 1

    return success_count + skipped, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Google Images Scraper - Download images for data gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python bantai_scrape.py "golden retriever" --count 50\n'
            '  python bantai_scrape.py "street signs" -n 200 -o ./dataset/signs\n'
            '  python bantai_scrape.py "cat" -n 500 --browser --visible\n'
            "\n"
            "To get more images for the same topic, run multiple queries into\n"
            "the same output folder with -o:\n"
            '  python bantai_scrape.py "overloaded sockets" -n 200 -o ./data --browser --visible\n'
            '  python bantai_scrape.py "overloaded power outlets" -n 200 -o ./data --browser --visible\n'
            "Already-downloaded images are skipped automatically.\n"
        ),
    )
    parser.add_argument("query", help="Search query for Google Images")
    parser.add_argument(
        "-n", "--count", type=int, default=50,
        help="Number of images to download (default: 50)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory (default: ./scraped_images/<query>)"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=8,
        help="Number of parallel download threads (default: 8)"
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="Use Selenium browser for more results (needs: pip install selenium)"
    )
    parser.add_argument(
        "--visible", action="store_true",
        help="Show browser window (useful to bypass CAPTCHA manually)"
    )
    parser.add_argument(
        "--min-res", type=str, default=None,
        help="Minimum resolution WxH to keep (e.g. 200x200). Requires Pillow."
    )
    parser.add_argument(
        "--max-res", type=str, default=None,
        help="Maximum resolution WxH to keep (e.g. 1920x1080). Requires Pillow."
    )

    args = parser.parse_args()

    # Parse resolution filters
    min_res = (0, 0)
    max_res = (0, 0)
    if args.min_res:
        try:
            parts = args.min_res.lower().split("x")
            min_res = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            print("[!] Invalid --min-res format. Use WxH (e.g. 200x200)")
            sys.exit(1)
    if args.max_res:
        try:
            parts = args.max_res.lower().split("x")
            max_res = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            print("[!] Invalid --max-res format. Use WxH (e.g. 1920x1080)")
            sys.exit(1)
    if (min_res != (0, 0) or max_res != (0, 0)) and not HAS_PIL:
        print("[!] Resolution filtering requires Pillow: pip install Pillow")
        sys.exit(1)

    # Default output directory
    if args.output is None:
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in args.query)
        safe_name = safe_name.strip().replace(" ", "_")
        args.output = os.path.join(".", "scraped_images", safe_name)

    print("=" * 60)
    print("  Google Images Scraper")
    print("=" * 60)
    print(f"  Query   : {args.query}")
    print(f"  Count   : {args.count}")
    print(f"  Output  : {args.output}")
    print(f"  Mode    : {'Browser (Selenium)' if args.browser else 'Requests (no browser)'}{'  [VISIBLE]' if args.visible else ''}")
    if min_res != (0, 0):
        print(f"  Min Res : {min_res[0]}x{min_res[1]}")
    if max_res != (0, 0):
        print(f"  Max Res : {max_res[0]}x{max_res[1]}")
    print("=" * 60)
    print()

    # Scrape URLs
    if args.browser:
        if not HAS_SELENIUM:
            print("[!] Selenium not installed. Install it with:")
            print("    pip install selenium")
            print("[*] Falling back to requests mode...")
            urls = scrape_with_requests(args.query, args.count)
        else:
            urls = scrape_with_selenium(args.query, args.count, visible=args.visible)
    else:
        urls = scrape_with_requests(args.query, args.count)

    if not urls:
        print("\n[!] No image URLs found. Try using --browser mode.")
        sys.exit(1)

    # Download
    success, failed = download_images(urls, args.output, workers=args.workers, min_res=min_res, max_res=max_res)

    # Summary
    print()
    print("=" * 60)
    print(f"  Done!")
    print(f"  Downloaded : {success}/{len(urls)} images")
    print(f"  Failed     : {failed}")
    print(f"  Saved to   : {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

