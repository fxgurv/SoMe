import os
import sys
import json
import srt_equalizer
from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])

def assert_folder_structure() -> None:
    if not os.path.exists(os.path.join(ROOT_DIR, "tmp")):
        if get_verbose():
            print(colored(f"=> Creating tmp folder at {os.path.join(ROOT_DIR, 'tmp')}", "green"))
        os.makedirs(os.path.join(ROOT_DIR, "tmp"))

def get_first_time_running() -> bool:
    return not os.path.exists(os.path.join(ROOT_DIR, "tmp"))

def get_email_credentials() -> dict:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["email"]

def get_verbose() -> bool:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["verbose"]

def get_firefox_profile_path() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["firefox_profile"]

def get_headless() -> bool:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["headless"]

def get_model() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["llm"]

def get_twitter_language() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["twitter_language"]

def get_image_model() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["image_model"]

def get_threads() -> int:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["threads"]
    
def get_image_prompt_llm() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["image_prompt_llm"]

def get_zip_url() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["zip_url"]

def get_is_for_kids() -> bool:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["is_for_kids"]

def get_google_maps_scraper_zip_url() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["google_maps_scraper"]

def get_google_maps_scraper_niche() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["google_maps_scraper_niche"]

def get_scraper_timeout() -> int:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["scraper_timeout"] or 300

def get_outreach_message_subject() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["outreach_message_subject"]
    
def get_outreach_message_body_file() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["outreach_message_body_file"]

def get_assemblyai_api_key() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["assembly_ai_api_key"]

def get_redis_uri() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["redis_uri"]

def get_user_data() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["user_data"]
    
def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
    
def get_font() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["font"]

def get_fonts_dir() -> str:
    return os.path.join(ROOT_DIR, "fonts")

def get_imagemagick_path() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["imagemagick_path"]


def get_tts_type() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("tts_type")


def get_elevenlabs_api_key() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["elevenlabs_api_key"]


def get_elevenlabs_voice() -> str:
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file).get("elevenlabs_voice")
