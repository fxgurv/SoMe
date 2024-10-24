import os
import json
from typing import List
from config import ROOT_DIR



def get_cache_path() -> str:
    return os.path.join(ROOT_DIR, '.mp')

def get_afm_cache_path() -> str:
    return os.path.join(get_cache_path(), 'afm.json')

def get_twitter_cache_path() -> str:
    return os.path.join(get_cache_path(), 'twitter.json')

def get_youtube_cache_path() -> str:
    return os.path.join(get_cache_path(), 'youtube.json')

def get_accounts(provider: str) -> List[dict]:
    cache_path = ""

    if provider == "twitter":
        cache_path = get_twitter_cache_path()
    elif provider == "youtube":
        cache_path = get_youtube_cache_path()

    if not os.path.exists(cache_path):
        with open(cache_path, 'w') as file:
            json.dump({
                "accounts": []
            }, file, indent=4)

    with open(cache_path, 'r') as file:
        parsed = json.load(file)

        if parsed is None:
            return []
        
        if 'accounts' not in parsed:
            return []

        return parsed['accounts']

def add_account(provider: str, account: dict) -> None:
    if provider == "twitter":
        accounts = get_accounts("twitter")
        accounts.append(account)
        with open(get_twitter_cache_path(), 'w') as file:
            json.dump({
                "accounts": accounts
            }, file, indent=4)
    elif provider == "youtube":
        accounts = get_accounts("youtube")
        accounts.append(account)
        with open(get_youtube_cache_path(), 'w') as file:
            json.dump({
                "accounts": accounts
            }, file, indent=4)

def remove_account(account_id: str) -> None:
    accounts = get_accounts()
    accounts = [account for account in accounts if account['id'] != account_id]
    with open(get_twitter_cache_path(), 'w') as file:
        json.dump({
            "accounts": accounts
        }, file, indent=4)

def get_products() -> List[dict]:
    if not os.path.exists(get_afm_cache_path()):
        with open(get_afm_cache_path(), 'w') as file:
            json.dump({
                "products": []
            }, file, indent=4)

    with open(get_afm_cache_path(), 'r') as file:
        parsed = json.load(file)
        return parsed["products"]
    
def add_product(product: dict) -> None:
    products = get_products()
    products.append(product)
    with open(get_afm_cache_path(), 'w') as file:
        json.dump({
            "products": products
        }, file, indent=4)
    
def get_results_cache_path() -> str:
    return os.path.join(get_cache_path(), 'scraper_results.csv')
