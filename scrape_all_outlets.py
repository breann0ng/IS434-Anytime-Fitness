"""
Standalone scraper that reuses the robust extraction logic applied to the notebook.

Usage examples:
  - Smoke test (only first outlet):
      python scrape_all_outlets.py --limit 1

  - Full run (ask me before running this):
      python scrape_all_outlets.py --limit 0   # 0 means no limit (all)

Notes:
- This script runs headless by default. It requires Chrome/Chromium and internet access.
- Output CSVs will be saved under `Reviews/All/` and filenames are based on outlet names.
"""

import os
import time
import argparse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR = os.getcwd()
GR_DIR = os.path.join(BASE_DIR, 'Google Reviews')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Reviews', 'All')


def safe_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()


def setup_driver(headless=True):
    options = ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--lang=en-US")
    options.add_argument("--window-size=1920,1080")
    if headless:
        # use new headless for recent Chrome
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    wait = WebDriverWait(driver, 20)
    return driver, wait


def scrape_reviews(outlet_name, outlet_url, driver, wait):
    """Scrape all reviews for a single outlet and save to CSV.
    Returns: dict with counts and path
    """
    print(f"\n--- Starting scrape for: {outlet_name}")
    try:
        driver.get(outlet_url)
    except Exception as e:
        return {"error": str(e)}

    time.sleep(2.5)

    try:
        reviews_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Reviews for")]'))
        )
        reviews_button.click()
    except Exception as e:
        return {"error": f"Could not click reviews button: {e}"}

    time.sleep(1.5)

    try:
        scrollable_div = wait.until(
            EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "m6QErb") and contains(@class, "DxyBCb")]'))
        )
    except Exception as e:
        return {"error": f"Could not find scrollable reviews container: {e}"}

    all_reviews_data = []
    seen_review_ids = set()
    no_text_rating_count = 0

    scroll_pause = 1.5
    no_new_count = 0
    max_no_new = 8
    previous_height = 0
    scroll_iteration = 0

    while True:
        scroll_iteration += 1

        review_elements = driver.find_elements(By.XPATH, '//div[@data-review-id]')
        new_reviews = [r for r in review_elements if r.get_attribute("data-review-id") not in seen_review_ids]

        if new_reviews:
            no_new_count = 0
            print(f"Iter {scroll_iteration}: found {len(new_reviews)} new reviews (total seen {len(seen_review_ids) + len(new_reviews)})")
        else:
            no_new_count += 1
            print(f"Iter {scroll_iteration}: no new reviews ({no_new_count}/{max_no_new})")
            if no_new_count >= max_no_new:
                print("Reached end of reviews for this outlet.")
                break

        for r in new_reviews:
            review_id = r.get_attribute("data-review-id")
            if review_id in seen_review_ids:
                continue
            seen_review_ids.add(review_id)

            try:
                # expand truncated
                try:
                    more_button = r.find_element(By.CLASS_NAME, 'w8nwRe')
                    driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(0.12)
                except (NoSuchElementException, StaleElementReferenceException):
                    pass

                # author
                try:
                    author_name = r.find_element(By.CLASS_NAME, 'd4r55').text
                except Exception:
                    author_name = ''

                rating_elements = r.find_elements(By.CLASS_NAME, 'kvMYJc')
                if not rating_elements:
                    # skip owner response
                    continue
                rating_element = rating_elements[0]
                rating_text = rating_element.get_attribute('aria-label') or ''
                try:
                    star_rating = int(rating_text.split(' ')[0])
                except Exception:
                    star_rating = None

                # robust extract text
                text_elems = r.find_elements(By.CLASS_NAME, 'wiI7pd')
                if text_elems:
                    review_text = (text_elems[0].text or '').strip()
                else:
                    review_text = ''

                if review_text == '':
                    no_text_rating_count += 1

                try:
                    date_element = r.find_element(By.CLASS_NAME, 'rsqApe')
                    posting_date = date_element.text
                except Exception:
                    posting_date = ''

                all_reviews_data.append({
                    'outlet': outlet_name,
                    'author': author_name,
                    'rating': star_rating,
                    'text': review_text,
                    'date_posted': posting_date,
                    'review_id': review_id
                })

            except Exception:
                continue

        # scroll small steps
        for _ in range(3):
            driver.execute_script("arguments[0].scrollBy(0, arguments[0].scrollHeight/3);", scrollable_div)
            time.sleep(0.25)
        time.sleep(scroll_pause)

        new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        if new_height == previous_height and not new_reviews:
            no_new_count += 1
        else:
            no_new_count = 0
        previous_height = new_height

        if scroll_iteration % 10 == 0:
            driver.execute_script("arguments[0].scrollBy(0, -400);", scrollable_div)
            time.sleep(0.3)

    # save
    if all_reviews_data:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = safe_filename(outlet_name) + '_reviews.csv'
        outpath = os.path.join(OUTPUT_DIR, fname)
        df = pd.DataFrame(all_reviews_data)
        df = df.drop_duplicates(subset=['author', 'text'], keep='first')
        df.to_csv(outpath, index=False)
        return {
            'outfile': outpath,
            'total_scraped': len(all_reviews_data),
            'unique_saved': len(df),
            'no_text_rating_count': no_text_rating_count
        }
    else:
        return {'outfile': None, 'total_scraped': 0, 'unique_saved': 0, 'no_text_rating_count': 0}


def main(limit=1):
    # Gather outlets from the provided master CSV
    final_csv = os.path.join(GR_DIR, 'final_SG_AFoutlets_ratings.csv')
    if os.path.exists(final_csv):
        try:
            df_outlets = pd.read_csv(final_csv)
        except Exception as e:
            print(f"Failed to read {final_csv}: {e}")
            return
    else:
        print('No outlet CSV found at Google Reviews/final_SG_AFoutlets_ratings.csv. Aborting.')
        return

    print(f"Found {len(df_outlets)} outlets to scrape (from final_SG_AFoutlets_ratings.csv).")

    driver, wait = setup_driver(headless=True)
    try:
        n = 0
        for idx, row in df_outlets.iterrows():
            n += 1
            if limit and limit > 0 and n > limit:
                break
            name = row.get('name')
            url = row.get('maps_url')
            if not name or not url:
                print(f"Skipping row {idx}: missing name or url")
                continue
            result = scrape_reviews(name, url, driver, wait)
            print(f"Result for {name}: {result}")
    finally:
        driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1, help='Number of outlets to scrape (0 = no limit)')
    args = parser.parse_args()
    lim = args.limit if args.limit != 0 else 0
    main(limit=lim)
