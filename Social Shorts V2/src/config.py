import os
import sys
import json
import srt_equalizer
from termcolor import colored

# Constants
ROOT_DIR = os.path.dirname(sys.path[0])
CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")

# Helper functions
def load_config():
    with open(CONFIG_FILE, "r") as file:
        return json.load(file)

def get_config_value(key, default=None):
    config = load_config()
    return config.get(key, default)

# Folder structure functions
def assert_folder_structure() -> None:
    mp_folder = os.path.join(ROOT_DIR, ".mp")
    if not os.path.exists(mp_folder):
        if get_verbose():
            print(colored(f"=> Creating .mp folder at {mp_folder}", "green"))
        os.makedirs(mp_folder)

def get_first_time_running() -> bool:
    return not os.path.exists(os.path.join(ROOT_DIR, ".mp"))

# Configuration getters
def get_verbose() -> bool:
    return get_config_value("verbose")

def get_headless() -> bool:
    return get_config_value("headless")

def get_browser_profile_path() -> str:
    return get_config_value("profile_path")

def get_browser() -> str:
    return get_config_value("browser")

def get_language() -> str:
    return get_config_value("language")

def get_threads() -> int:
    return get_config_value("threads")

def get_is_for_kids() -> bool:
    return get_config_value("is_for_kids")

def get_scraper_timeout() -> int:
    return get_config_value("scraper_timeout", 300)

def get_font() -> str:
    return get_config_value("font")

def get_fonts_dir() -> str:
    return os.path.join(ROOT_DIR, "fonts")

def get_imagemagick_path() -> str:
    return get_config_value("imagemagick_path")

# Model and LLM related functions
def get_model() -> str:
    return get_config_value("llm")

def get_image_model() -> str:
    return get_config_value("image_model")

def get_image_prompt_llm() -> str:
    return get_config_value("image_prompt_llm")

def get_tts_engine() -> str:
    return get_config_value("tts_engine")

def get_tts_voice() -> str:
    return get_config_value("tts_voice")

# API Keys (sorted alphabetically)
def get_assemblyai_api_key() -> str:
    return get_config_value("assembly_ai_api_key")

def get_elevenlabs_api_key() -> str:
    return get_config_value("elevenlabs_api_key")

def get_gemini_api_key() -> str:
    return get_config_value("gemini_api_key")

def get_openai_api_key() -> str:
    return get_config_value("openai_api_key")

# Email and messaging functions
def get_email_credentials() -> dict:
    return get_config_value("email")

def get_outreach_message_subject() -> str:
    return get_config_value("outreach_message_subject")

def get_outreach_message_body_file() -> str:
    return get_config_value("outreach_message_body_file")

# Scraper and URL related functions
def get_zip_url() -> str:
    return get_config_value("zip_url")

def get_google_maps_scraper_zip_url() -> str:
    return get_config_value("google_maps_scraper")

def get_google_maps_scraper_niche() -> str:
    return get_config_value("google_maps_scraper_niche")

# Telegram related functions
def get_telegram_api_id():
    return get_config_value("telegram_api_id")

def get_telegram_api_hash():
    return get_config_value("telegram_api_hash")

def get_phone():
    return get_config_value("phone_number")

def get_db_file():
    return get_config_value("db_file")

# Subtitle related function
def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
