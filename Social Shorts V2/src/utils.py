import os
import random
import zipfile
import requests
import platform
from status import *
from config import *

def close_running_selenium_instances() -> None:
    try:
        info(" => Closing running Selenium instances...")

        if platform.system() == "Windows":
            os.system("taskkill /f /im firefox.exe")
        else:
            os.system("pkill firefox")

        success(" => Closed running Selenium instances.")

    except Exception as e:
        error(f"Error occurred while closing running Selenium instances: {str(e)}")

def build_url(youtube_video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_video_id}"

def rem_temp_files() -> None:
    mp_dir = os.path.join(ROOT_DIR, ".mp")
    files = os.listdir(mp_dir)

    for file in files:
        if not file.endswith(".json"):
            os.remove(os.path.join(mp_dir, file))

def fetch_Music() -> None:
    try:
        info(f" => Fetching Music...")

        files_dir = os.path.join(ROOT_DIR, "Music")
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
            if get_verbose():
                info(f" => Created directory: {files_dir}")
        else:
            return

        response = requests.get(get_zip_url() or "https://filebin.net/bb9ewdtckolsf3sg/drive-download-20240209T180019Z-001.zip")

        with open(os.path.join(files_dir, "Music.zip"), "wb") as file:
            file.write(response.content)

        with zipfile.ZipFile(os.path.join(files_dir, "Music.zip"), "r") as file:
            file.extractall(files_dir)

        os.remove(os.path.join(files_dir, "Music.zip"))

        success(" => Downloaded Music to ../Music.")

    except Exception as e:
        error(f"Error occurred while fetching Music: {str(e)}")

def choose_random_music() -> str:
    try:
        music_files = os.listdir(os.path.join(ROOT_DIR, "Music"))
        chosen_music = random.choice(music_files)
        success(f"Successfully chose random background Music: {chosen_music}")
        return os.path.join(ROOT_DIR, "Music", chosen_music)
    except Exception as e:
        error(f"Error occurred while choosing random Music: {str(e)}")
