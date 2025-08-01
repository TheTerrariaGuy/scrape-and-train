import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import time

total = []
allurls = [] 

def gettext(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded)

def geturls(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return []
    soup = BeautifulSoup(downloaded, 'html.parser')
    urls = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        absolute_url = urljoin(url, href)
        if urlparse(absolute_url).scheme in ['http', 'https']:
            urls.append(absolute_url)
    return urls

def recurse_urls(url, depth):
    if depth == 0:
        return []
    
    print(f"Processing: {url} (depth: {depth})")
    text = gettext(url)
    if text:
        total.append(text)
    urls = geturls(url)
    all_urls = urls[:]
    for u in urls[:5]:
        try:
            all_urls.extend(recurse_urls(u, depth - 1))
        except Exception as e:
            print(f"Error processing {u}: {e}")
            continue
    
    return all_urls

def scrape_urls_from_csv(csv_file, max_urls=None):
    df = pd.read_csv(csv_file)
    if 'link' not in df.columns:
        raise ValueError("CSV file must have a 'link' column")
    if max_urls:
        df = df.head(max_urls)

    results = []
    for index, row in df.iterrows():
        url = row['link']
        name = row.get('name', f"URL_{index}")
        # print(f"curr {index + 1}/{len(df)}: {name}")
        
        
        try:
            time.sleep(0.5)
            text = gettext(url)
            content = text if text else ""
            
            results.append({
                'name': name,
                'url': url,
                'content': content,
                'success': bool(text)
            })
        except Exception as e:
            print(f"Error processing {url}: {e}")
            results.append({
                'name': name,
                'url': url,
                'content': "",
                'success': False
            })
    return pd.DataFrame(results)

def main():
    csv_file = 'out.csv'
    try:
        results_df = scrape_urls_from_csv(csv_file, max_urls=None)
        output_file = 'scraped_content.feather'
        results_df.to_feather(output_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    
