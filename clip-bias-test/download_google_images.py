import os, urllib.request, json # json for pretty output
from serpapi import GoogleSearch
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(description='Download Google Images')
parser.add_argument('--query', type=str, required=True, help='Query to download')
args = parser.parse_args()

start = timer()

# query: a photo of a female, a photo of male
def get_google_images():
    params = {
      "api_key": "aca86d214a841b7daa52db6a635b0156c773dd2ffb9068373bf521a6d5a819c5",
      "engine": "google",
      "q": args.query,
      "tbm": "isch"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # print(json.dumps(results['suggested_searches'], indent=2, ensure_ascii=False))
    print(json.dumps(results['images_results'], indent=2, ensure_ascii=False))

    if(not os.path.isdir(f'test/{args.query}')):
        os.makedirs(f'test/{args.query}')

    # -----------------------
    # Downloading images

    for index, image in enumerate(results['images_results']):

        print(f'Downloading {index} image...')
        
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(image['original'], f'test/{args.query}/{args.query}_{index}.jpg')

    end = timer()
    print(end - start)

if __name__ == '__main__':
	get_google_images()