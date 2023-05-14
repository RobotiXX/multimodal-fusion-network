import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By


wget = []
label = []
processed = []
un_processed = []

def process_link(link):
    # Configure Selenium WebDriver
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode (without opening a browser window)
        driver = webdriver.Chrome(options=options)

        # Open the link
        driver.get(link)
        time.sleep(2)  # Wait for page to load (adjust as needed)

        # Find the text inside the <code> tag
        code_element = driver.find_element(By.TAG_NAME, 'code')
        code_text = code_element.text
        print("Text inside <code> tag:", code_text)
       
        # Find the element with id "fileDescriptionBlock" ̏
        description_element = driver.find_element(By.ID, 'fileDescriptionBlock')

        # Find the <td> element within the description element and extract the text inside it
        td_element = description_element.find_element(By.TAG_NAME, 'td')
        td_text = td_element.text
        
        wget.append(code_text)
        label.append(td_text)
        processed.append(link)
        print("Text inside <td> element:", td_text)

        # Close the browser
        driver.quit()
    except:
        print(f"===>>>>could not load {link}")
        un_processed.append(link)

# Example usage
links = [
    "https://dataverse.tdl.org/file.xhtml?fileId=136510&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138181&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138164&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=133251&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136672&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138166&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=135968&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136021&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138265&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136679&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138139&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138143&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136455&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138258&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136685&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=133406&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136459&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138186&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138158&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138168&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138142&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138157&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136514&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136566&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136567&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138167&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=133228&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138173&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136453&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138263&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138260&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138262&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138141&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138144&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138153&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138146&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138147&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136498&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138184&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138174&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136500&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136515&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=135967&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136680&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138138&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138163&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136450&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136511&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138264&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136564&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138185&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138261&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136682&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138148&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136683&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136015&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136681&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138133&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138191&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136017&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138187&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=136671&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138132&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138152&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138145&version=4.0",
    "https://dataverse.tdl.org/file.xhtml?fileId=138160&version=4.0"
]

print(f"length: {len(links)}")
counter = 1
for link in links:
    print(f" counter: {counter} link: {link}")
    process_link(link)
    print("-----")
    counter += 1

# print(wget)
# print(label)

# Create a list of tuples by merging the arrays side by side
merged_data = list(zip(processed, wget, label))

# Specify the output CSV file path
output_csv_file = 'merged_data.csv'

# Write the merged data to the CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Web Link','Download Link', 'Label'])  # Write the header row
    writer.writerows(merged_data)  # Write the merged data rows


print(un_processed)