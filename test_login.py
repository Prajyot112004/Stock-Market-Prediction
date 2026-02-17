import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# 1. Setup Headless Chrome (Required for Jenkins)
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)

try:
    # 2. Go to the app
    driver.get("http://localhost:8501")
    
    # 3. THE FIX: Wait longer for Streamlit to load the UI
    print("Waiting for Streamlit UI to render...")
    time.sleep(15) 
    
    # 4. Check if we are on the right page
    # Look for a common Streamlit element or your specific title
    if "Stock" in driver.title or len(driver.find_elements(By.TAG_NAME, "button")) > 0:
        print("SUCCESS: Selenium Test Passed! App is live and showing content.")
    else:
        print("ERROR: The robot saw a blank page.")
        exit(1) # This tells Jenkins the test failed

finally:
    driver.quit()
