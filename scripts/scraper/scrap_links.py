import time
from selenium import webdriver
from selenium.webdriver.common.by import By

def get_links_with_text(driver, url, start_text):
    # Open the URL
    driver.get(url)
    time.sleep(2)  # Wait for page to load (adjust as needed)
    
    # Find and click the button with the specified text
    button = driver.find_element(By.XPATH, f"//span[contains(text(), '{start_text}')]")
    button.click()
    time.sleep(10)  # Wait for button action to take effect (adjust as needed)

    # Extract the links
    links = []
    elements = driver.find_elements(By.TAG_NAME, 'a')
    for element in elements:
        if element.text.startswith('A_Spot'): 
            href = element.get_attribute('href')
            links.append(href)

    return links

# Example usage
url = 'https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH&version=4.0'
start_text = 'Tree'

# Configure Selenium WebDriver
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Run in headless mode (without opening a browser window)
driver = webdriver.Chrome(options=options)

links = get_links_with_text(driver, url, start_text)
for link in links:
    print(link)

# Close the browser
driver.quit()
