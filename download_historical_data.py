import os
import requests
import datetime
from bs4 import BeautifulSoup
import time

# List provided by user
data_list = """
EUR/USD,May 2000
EUR/CHF,May 2000
EUR/GBP,March 2002
EUR/JPY,March 2002
EUR/AUD,August 2002
USD/CAD,June 2000
USD/CHF,May 2000
USD/JPY,May 2000
USD/MXN,November 2010
GBP/CHF,August 2002
GBP/JPY,May 2002
GBP/USD,May 2000
AUD/JPY,August 2002
AUD/USD,June 2000
CHF/JPY,August 2002
NZD/JPY,September 2006
NZD/USD,August 2005
XAU/USD,March 2009
EUR/CAD,March 2007
AUD/CAD,July 2007
CAD/JPY,March 2007
EUR/NZD,March 2008
GRX/EUR,November 2010
NZD/CAD,March 2008
SGD/JPY,August 2008
USD/HKD,August 2008
USD/NOK,August 2008
USD/TRY,November 2010
XAU/AUD,May 2009
AUD/CHF,March 2008
AUX/AUD,November 2010
EUR/HUF,November 2010
EUR/PLN,November 2010
FRX/EUR,November 2010
HKX/HKD,November 2010
NZD/CHF,March 2008
SPX/USD,November 2010
USD/HUF,November 2010
USD/PLN,November 2010
USD/ZAR,November 2010
XAU/CHF,May 2009
ZAR/JPY,November 2010
BCO/USD,November 2010
ETX/EUR,November 2010
EUR/CZK,November 2010
EUR/SEK,August 2008
GBP/AUD,September 2007
GBP/NZD,March 2008
JPX/JPY,November 2010
UDX/USD,November 2010
USD/CZK,August 2010
USD/SEK,August 2008
WTI/USD,November 2010
XAU/EUR,May 2009
AUD/NZD,September 2007
CAD/CHF,March 2008
EUR/DKK,August 2008
EUR/NOK,August 2008
EUR/TRY,November 2010
GBP/CAD,September 2007
NSX/USD,November 2010
UKX/GBP,November 2010
USD/DKK,August 2008
USD/SGD,August 2008
XAG/USD,May 2009
XAU/GBP,May 2009
"""

BASE_URL = "https://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes/"
POST_URL = "https://www.histdata.com/get.php"
DOWNLOAD_DIR = "forex_data"

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def download_pair_year(pair_slug, year):
    target_url = f"{BASE_URL}{pair_slug}/{year}"
    session = requests.Session()
    
    try:
        # Step 1: Get the page to extract form hidden values and session cookie
        response = session.get(target_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        form = soup.find('form', {'id': 'file_down'})
        
        if not form:
            print(f"No download form found for {pair_slug} {year}")
            return

        # Extract hidden inputs (tk, id, date, datemonth)
        payload = {input_tag.get('name'): input_tag.get('value') 
                   for input_tag in form.find_all('input') 
                   if input_tag.get('name')}

        # Step 2: POST to get the file
        # The site expects a referer and the specific hidden fields
        headers = {'Referer': target_url}
        file_res = session.post(POST_URL, data=payload, headers=headers, stream=True)
        
        if file_res.status_code == 200:
            # Get filename from headers or default
            disp = file_res.headers.get('Content-Disposition')
            filename = disp.split('filename=')[1].strip('"') if disp else f"{pair_slug}_{year}.zip"
            
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            with open(filepath, 'wb') as f:
                for chunk in file_res.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded: {filename}")
        else:
            print(f"Failed to download {pair_slug} {year} (Status: {file_res.status_code})")

    except Exception as e:
        print(f"Error processing {pair_slug} {year}: {e}")

# Main execution logic
current_year = datetime.datetime.now().year
lines = [line.strip() for line in data_list.strip().split('\n') if line.strip()]

for line in lines[1:]:  # Skip header
    pair_raw, start_date_str = line.split(',')
    
    # Format pair: EUR/USD -> eurusd
    pair_slug = pair_raw.replace('/', '').lower()
    
    # Extract year: "May 2000" -> 2000
    start_year = int(start_date_str.split(' ')[-1])
    
    print(f"Starting downloads for {pair_raw} (From {start_year} to {current_year})")
    
    for year in range(start_year, current_year + 1):
        download_pair_year(pair_slug, year)
        # Be polite to the server
        time.sleep(1)