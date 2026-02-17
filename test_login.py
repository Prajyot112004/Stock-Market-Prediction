from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 1. Setup the invisible browser
options = Options()
options.add_argument("--headless") 
driver = webdriver.Chrome(options=options)

# 2. Go to your app's house (localhost)
driver.get("http://localhost:8501")

# 3. Look for the word "Stock"
if "Stock" in driver.page_source:
    print("SUCCESS: The robot saw the Stock App!")
else:
    print("ERROR: The robot saw a blank page.")
    exit(1) # This tells the Factory Manager something is wrong

driver.quit()