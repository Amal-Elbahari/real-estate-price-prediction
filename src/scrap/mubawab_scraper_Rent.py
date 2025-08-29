from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
import psycopg2

# Fonction pour créer un nouveau driver Chrome (headless)
def create_driver():
    chrome_options = Options()
   # chrome_options.add_argument("--headless=new")  # mode headless stable
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--start-maximized")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(120)
    return driver

driver = create_driver()

base_urls = {
    "For Rent": "https://www.mubawab.ma/en/cc/real-estate-for-rent"
}

def clean_text(text):
    if not text:
        return None
    return text.strip().replace('\n', ' ').replace('\t', ' ')

def scrape_property(property_element, url_type='For Sale'):
    data = {
        'title': None, 'price': None, 'location': None, 'description': None,
        'surface': None, 'rooms': None, 'bedrooms': None, 'bathrooms': None,
        'type': url_type, 'features': None, 'property_category': None,
        'link': None, 'image_url': None, 'published_time': None
    }

    try:
        title_element = property_element.find_element(By.CSS_SELECTOR, 'h2.listingTit a')
        data['title'] = clean_text(title_element.text)
        data['link'] = title_element.get_attribute('href')
        title_lower = data['title'].lower() if data['title'] else ''
        if "apartment" in title_lower:
            data['property_category'] = "Apartment"
        elif "villa" in title_lower:
            data['property_category'] = "Villa"
        elif "house" in title_lower:
            data['property_category'] = "House"
        else:
            data['property_category'] = "Autre"
    except NoSuchElementException:
        pass

    try:
        data['price'] = clean_text(property_element.find_element(By.CLASS_NAME, 'priceTag').text)
    except NoSuchElementException:
        pass

    try:
        data['location'] = clean_text(property_element.find_element(By.CSS_SELECTOR, '.listingH3').text)
    except NoSuchElementException:
        pass

    try:
        data['surface'] = clean_text(property_element.find_element(By.CSS_SELECTOR, '.icon-triangle + span').text)
    except NoSuchElementException:
        pass

    try:
        data['rooms'] = clean_text(property_element.find_element(By.CSS_SELECTOR, '.icon-house-boxes + span').text)
    except NoSuchElementException:
        pass

    try:
        data['bedrooms'] = clean_text(property_element.find_element(By.CSS_SELECTOR, '.icon-bed + span').text)
    except NoSuchElementException:
        pass

    try:
        data['bathrooms'] = clean_text(property_element.find_element(By.CSS_SELECTOR, '.icon-bath + span').text)
    except NoSuchElementException:
        pass

    try:
        features = property_element.find_elements(By.CSS_SELECTOR, '.adFeature span')
        features_texts = [clean_text(feat.text) for feat in features if clean_text(feat.text)]
        data['features'] = ", ".join(features_texts) if features_texts else None
    except NoSuchElementException:
        pass

    try:
        data['published_time'] = clean_text(property_element.find_element(By.CSS_SELECTOR, 'p.card-footer__publish-date').text)
    except NoSuchElementException:
        pass

    try:
        img_element = property_element.find_element(By.CSS_SELECTOR, 'div.photoBox img.w100')
        img_src = img_element.get_attribute('src') or img_element.get_attribute('data-lazy')
        data['image_url'] = img_src
    except NoSuchElementException:
        pass

    try:
        desc_element = property_element.find_element(By.CSS_SELECTOR, 'h2.listingTit a')
        data['description'] = clean_text(desc_element.text)
    except NoSuchElementException:
        pass

    return data

def load_page_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Tentative {attempt + 1}: Chargement de {url}")
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "listingBox")))
            time.sleep(3)
            return True
        except (TimeoutException, WebDriverException) as e:
            print(f"Tentative {attempt + 1} échouée: {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(5)
    return False

def scrape_page(url, url_type):
    try:
        if not load_page_with_retry(url):
            print(f"Échec du chargement de la page {url}")
            return []
        if "captcha" in driver.page_source.lower():
            print("CAPTCHA détecté !")
            return []
        try:
            driver.find_element(By.CLASS_NAME, 'noResultsTxt')
            print("Aucun résultat trouvé")
            return []
        except NoSuchElementException:
            pass
        properties = driver.find_elements(By.CLASS_NAME, 'listingBox')
        return [scrape_property(p, url_type) for p in properties if p.is_displayed()]
    except WebDriverException as e:
        # Propagation de l'erreur pour relancer le driver
        raise e
    except Exception as e:
        print(f"Erreur lors du scraping: {str(e)}")
        return []

def save_to_csv(data, filename='mubawab_rent.csv'):
    if not data or all(not any(d.values()) for d in data):
        print("Aucune donnée valide à sauvegarder.")
        return

    fieldnames = [
        'title', 'price', 'location', 'description',
        'surface', 'rooms', 'bedrooms', 'bathrooms',
        'type', 'features', 'property_category', 'link',
        'image_url', 'published_time'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([d for d in data if any(d.values())])

    print(f"Données sauvegardées dans {filename}")

def insert_csv_to_postgres(csv_file, dbname, user, password, host='localhost', port='5432'):
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cursor = conn.cursor()

    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mubawab_rent(
            id SERIAL PRIMARY KEY,
            title TEXT, price TEXT, location TEXT, description TEXT,
            surface TEXT, rooms TEXT, bedrooms TEXT, bathrooms TEXT,
            type TEXT, features TEXT, property_category TEXT,
            link TEXT, image_url TEXT, published_time TEXT
        );
    """)
    conn.commit()

    
    

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [
            (
                row['title'], row['price'], row['location'], row['description'],
                row['surface'], row['rooms'], row['bedrooms'], row['bathrooms'],
                row['type'], row['features'], row['property_category'],
                row['link'], row['image_url'], row['published_time']
            )
            for row in reader
        ]

    insert_query = """
        INSERT INTO mubawab_rent (
            title, price, location, description,
            surface, rooms, bedrooms, bathrooms,
            type, features, property_category,
            link, image_url, published_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"✅ {len(rows)} lignes insérées dans PostgreSQL.")

    cursor.close()
    conn.close()

def main():
    global driver
    all_data = []
    pages_to_scrape = 200

    for page in range(201, pages_to_scrape + 1):
        url = f"{base_urls['For Rent']}?page={page}"
        print(f"\n📄 Scraping page {page}/{pages_to_scrape}...")

        try:
            page_data = scrape_page(url, "For Rent")
            if page_data:
                all_data.extend(page_data)
                print(f"    ✅ {len(page_data)} propriétés extraites")
            else:
                print("    ⚠️ Aucune donnée sur cette page")
            time.sleep(3)
        except WebDriverException as e:
            print(f"Erreur WebDriver détectée: {e}")
            print("Relance du driver et reprise du scraping...")
            try:
                driver.quit()
            except Exception:
                pass
            driver = create_driver()
            # Re-tenter la page en cours après relance du driver
            try:
                page_data = scrape_page(url, "For Rent")
                if page_data:
                    all_data.extend(page_data)
                    print(f"    ✅ {len(page_data)} propriétés extraites (après relance)")
                else:
                    print("    ⚠️ Aucune donnée sur cette page (après relance)")
            except Exception as e2:
                print(f"Nouvelle erreur après relance: {e2}")
            time.sleep(3)

    if all_data:
        save_to_csv(all_data)
        insert_csv_to_postgres(
            csv_file='mubawab_rent.csv',
            dbname='real',
            user='postgres',
            password='2024',
            host='localhost',
            port='5432'
        )
        print("\n✅ Scraping + insertion PostgreSQL terminé !")
    else:
        print("\n⚠️ Aucune donnée récupérée.")

    try:
        driver.quit()
    except Exception:
        pass

if __name__ == "__main__":
    main()
