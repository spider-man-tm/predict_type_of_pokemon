from requests import exceptions
import argparse
import requests
import cv2
import os
import yaml
from utils import load_generation_yml


API_KEY = '******************************'
MAX_RESULTS = 6
GROUP_SIZE = 6
URL = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'
headers = {
    'Ocp-Apim-Subscription-Key' : API_KEY
    }
EXCEPTIONS = set([
    IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout
    ])

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_test', type=str, help='Choice Train or Test')
parser.add_argument('-g', '--generation', type=int, help='Choice Pokemon Generation')
args = parser.parse_args()
generation = args.generation
train_test = args.train_test


def mkdir_func(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def make_param(term):
    params = {
        'q': term,
        'offset': 0,
        'count': GROUP_SIZE,
        'imageType': 'Photo',
        'color': 'ColorOnly',
        'aspect': 'Square',
    }
    return params


def set_params(headers, params):
    print(f"\nLet's search for {params['q']}\n")
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    estNumResults = min(results['totalEstimatedMatches'], MAX_RESULTS)
    return estNumResults


def main():
    total = generation * 10000
    output = os.path.join(f'data/{train_test}')
    mkdir_func(output)
    pokemons = load_generation_yml(generation)
    for term, pokemon_type in pokemons.items():
        params = make_param(term)
        estNumResults = set_params(headers, params)
        for offset in range(0, estNumResults, GROUP_SIZE):
            params["offset"] = offset
            search = requests.get(URL, headers=headers, params=params)
            search.raise_for_status()
            results = search.json()

            for v in results["value"]:
                try:
                    print(f'[Fetch]  {v["contentUrl"]}')
                    r = requests.get(v['contentUrl'], timeout=30)
                    ext = v['contentUrl'][v['contentUrl'].rfind('.'):v['contentUrl'].rfind('?') if v['contentUrl'].rfind('?') > 0 else None]
                    
                    if ext=='.jpg' or ext=='.png':
                        if len(pokemon_type)==1:
                            p = os.path.sep.join([output, f'{str(total).zfill(5)}_{pokemon_type[0]}.jpg'])
                        elif len(pokemon_type)==2:
                            p = os.path.sep.join([output, f'{str(total).zfill(5)}_{pokemon_type[0]}_{pokemon_type[1]}.jpg'])
                        with open(p, 'wb') as f:
                            f.write(r.content)
                    else:
                        continue

                except Exception as e:
                    if type(e) in EXCEPTIONS:
                        print(f'[Skip]  {v["contentUrl"]}')
                        continue

                image = cv2.imread(p)

                if image is None:
                    print(f'[Delite]  {p}')
                    os.remove(p)
                    continue

                # update the counter
                total += 1


if __name__ == '__main__':
    main()