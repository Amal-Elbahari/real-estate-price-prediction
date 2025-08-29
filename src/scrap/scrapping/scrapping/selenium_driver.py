from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ⚙️ Options du navigateur
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless=new")  # headless mode + faster
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--no-sandbox")

def get_driver():
    """Returns a Selenium Chrome WebDriver using ChromeDriverManager."""
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
