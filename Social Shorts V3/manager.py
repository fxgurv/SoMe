import os
import sys
import platform
import shutil
from uuid import uuid4
from termcolor import colored
from dotenv import load_dotenv
from text import generate_all
from image import generate_images
from tts import TTS, convert_to_mp3
from moviepy.editor import *

# Load environment variables
load_dotenv('config.env')

# Constants
LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian",
    "Japanese", "Korean", "Chinese", "Arabic", "Hindi", "Dutch", "Swedish", "Polish"
]

OPERATING_SYSTEMS = [
    "Windows", "Linux", "macOS", "Android", "iOS", "FreeBSD", "OpenBSD", "Chrome OS",
    "Solaris", "AIX", "HP-UX", "z/OS", "Other"
]

LOGGING = True

# Beautiful logging functions
def log_message(message: str, color: str, emoji: str = "") -> None:
    if LOGGING:
        terminal_width = shutil.get_terminal_size().columns
        formatted_message = f"{emoji} {message}".center(terminal_width)
        print(colored(formatted_message, color))

def info(message: str): log_message(message, "cyan", "ℹ️")
def error(message: str): log_message(message, "red", "❌")
def success(message: str): log_message(message, "green", "✅")
def warning(message: str): log_message(message, "yellow", "⚠️")
def verify(message: str): log_message(message, "magenta", "🔍")
def true(message: str): log_message(message, "green", "✔️")
def false(message: str): log_message(message, "red", "❌")
def skip(message: str): log_message(message, "yellow", "⏭️")
def existing(message: str): log_message(message, "blue", "🔄")
def prompt(message: str) -> str: return input(colored(f"❓ {message}", "light_magenta"))
def minimum(message: str): log_message(message, "yellow", "⬇️")
def maximum(message: str): log_message(message, "yellow", "⬆️")

def determine_os():
    info("Determining operating system")
    system = platform.system()
    if system == "Windows":
        success("Detected Windows operating system")
        return "Windows"
    elif system == "Linux":
        if 'ANDROID_ROOT' in os.environ:
            success("Detected Android operating system")
            return "Android"
        success("Detected Linux operating system")
        return "Linux"
    elif system == "Darwin":
        success("Detected macOS operating system")
        return "macOS"
    else:
        warning("Unable to automatically detect your operating system")
        info("Please select your operating system from the list below:")
        for idx, os_name in enumerate(OPERATING_SYSTEMS, 1):
            info(f"{idx}. {os_name}")
        
        while True:
            choice = prompt("Enter the number of your operating system: ")
            if choice.isdigit() and 1 <= int(choice) <= len(OPERATING_SYSTEMS):
                selected_os = OPERATING_SYSTEMS[int(choice) - 1]
                success(f"Selected operating system: {selected_os}")
                return selected_os
            else:
                error("Invalid choice. Please try again.")

def print_banner():
    banner_path = os.path.join(os.getcwd(), "Documents", "Banner.txt")
    try:
        with open(banner_path, 'r') as banner_file:
            banner_content = banner_file.read().strip()
        terminal_width = shutil.get_terminal_size().columns
        banner_lines = banner_content.split('\n')
        centered_banner = '\n'.join(line.center(terminal_width) for line in banner_lines)
        print(centered_banner)
    except FileNotFoundError:
        error("Banner file not found. Please ensure 'Banner.txt' is in the Documents folder.")

def clear_screen(current_os):
    os.system('cls' if current_os == "Windows" else 'clear')

def print_menu(options):
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    print("0. Exit")

def get_user_input(prompt_message, default=None):
    user_input = prompt(prompt_message)
    return user_input if user_input else default

def create_account(current_os):
    info("Creating new account")
    account_name = get_user_input("Account Name: ", os.getenv('DEFAULT_ACCOUNT_NAME', 'Account 1'))
    name = get_user_input("Full Name: ", os.getenv('DEFAULT_NAME', 'Guest'))
    
    info("Available Languages:")
    for idx, lang in enumerate(LANGUAGES, 1):
        print(f"{idx}. {lang}")
    
    lang_choice = get_user_input("Select Language (number): ", os.getenv('DEFAULT_LANGUAGE', '1'))
    language = LANGUAGES[int(lang_choice) - 1] if lang_choice.isdigit() and 1 <= int(lang_choice) <= len(LANGUAGES) else "English"
    
    niche = get_user_input("Niche: ", os.getenv('DEFAULT_NICHE', 'Human History'))
    
    with open('account.env', 'w') as f:
        f.write(f"ACCOUNT_NAME={account_name}\n")
        f.write(f"NAME={name}\n")
        f.write(f"LANGUAGE={language}\n")
        f.write(f"NICHE={niche}\n")
        f.write(f"VIDEOS=[]\n")
        f.write(f"OPERATING_SYSTEM={current_os}\n")
    
    success("Account created successfully!")
    return account_name

def load_account():
    if not os.path.exists('account.env'):
        return None
    
    load_dotenv('account.env')
    account = {
        'ACCOUNT_NAME': os.getenv('ACCOUNT_NAME'),
        'NAME': os.getenv('NAME'),
        'LANGUAGE': os.getenv('LANGUAGE'),
        'NICHE': os.getenv('NICHE'),
        'VIDEOS': eval(os.getenv('VIDEOS', '[]')),
        'OPERATING_SYSTEM': os.getenv('OPERATING_SYSTEM')
    }
    return account

def add_video(account):
    video_title = get_user_input("Enter video title: ")
    video_path = get_user_input("Enter video file path: ")
    if os.path.exists(video_path):
        account['VIDEOS'].append({"title": video_title, "path": video_path})
        save_account(account)
        success("Video added successfully")
    else:
        error("Video file not found")

def get_videos(account):
    return account['VIDEOS']

def save_account(account):
    with open('account.env', 'w') as f:
        for key, value in account.items():
            f.write(f"{key}={value}\n")

def generate_video(account):
    info("Generating video content")
    content = generate_content(account['NICHE'], account['LANGUAGE'])
    
    image_folder = "temp_images"
    os.makedirs(image_folder, exist_ok=True)
    image_paths = generate_images(content['image_prompts'], image_folder)
    
    tts = TTS(engine='gtts', voice=account['LANGUAGE'][:2].lower())
    audio_file = "temp_audio.mp3"
    tts.synthesize(content['script'], audio_file)
    
    video_file = f"{content['topic'][:30]}.mp4"
    create_video(image_paths, audio_file, video_file)
    
    for image in image_paths:
        os.remove(image)
    os.remove(audio_file)
    os.rmdir(image_folder)
    
    success(f"Video generated: {video_file}")
    return video_file, content['metadata']

def create_video(image_paths, audio_file, output_file):
    audio = AudioFileClip(audio_file)
    duration = audio.duration
    
    clips = [ImageClip(m).set_duration(duration/len(image_paths)) for m in image_paths]
    concat_clip = concatenate_videoclips(clips, method="compose")
    final_clip = concat_clip.set_audio(audio)
    final_clip.write_videofile(output_file, fps=24)

def test_functions():
    info("Testing various functions")
    
    # Test text generation
    info("Testing text generation with OpenAI")
    openai_text = generate_content("Test topic", "English", engine="openai")
    success(f"OpenAI text generated: {openai_text[:100]}...")
    
    info("Testing text generation with Claude")
    claude_text = generate_content("Test topic", "English", engine="claude")
    success(f"Claude text generated: {claude_text[:100]}...")
    
    info("Testing text generation with G4F")
    g4f_text = generate_content("Test topic", "English", engine="g4f")
    success(f"G4F text generated: {g4f_text[:100]}...")
    
    # Test image generation
    info("Testing image generation with Hercai")
    hercai_image = generate_images(["A beautiful landscape"], "temp_images", engine="hercai")
    success(f"Hercai image generated: {hercai_image}")
    
    info("Testing image generation with Segmind")
    segmind_image = generate_images(["A futuristic city"], "temp_images", engine="segmind")
    success(f"Segmind image generated: {segmind_image}")
    
    # Test TTS
    info("Testing TTS")
    tts = TTS(engine='gtts', voice='en')
    tts.synthesize("This is a test of the text-to-speech system.", "test_tts.mp3")
    success("TTS test completed")
    
    info("All tests completed successfully")

def main_menu_logged_in(account, current_os):
    while True:
        clear_screen(current_os)
        print_banner()
        print(f"Welcome back, {account['NAME']}!")
        options = [
            "Generate Video",
            "View Videos",
            "Add Video",
            "My Account",
            "Test Functions",
            "User Manual"
        ]
        print_menu(options)
        
        choice = prompt("Select an option: ")
        
        if choice == '1':
            video_file, metadata = generate_video(account)
            account['VIDEOS'].append({"file": video_file, "metadata": metadata})
            save_account(account)
        elif choice == '2':
            videos = get_videos(account)
            if videos:
                print("Your Videos:")
                for idx, video in enumerate(videos, 1):
                    print(f"{idx}. {video['file']} - {video['metadata']['title']}")
            else:
                warning("No videos found.")
        elif choice == '3':
            add_video(account)
        elif choice == '4':
            # Add logic to change account settings
            pass
        elif choice == '5':
            test_functions()
        elif choice == '6':
            # Add logic to display user manual
            pass
        elif choice == '0':
            sys.exit(0)
        else:
            error("Invalid option selected")
        
        prompt("Press Enter to continue...")

def main_menu_logged_out(current_os):
    while True:
        clear_screen(current_os)
        print_banner()
        options = [
            "Sign up",
            "Guest Login",
            "Add Account",
            "User Manual"
        ]
        print_menu(options)
        
        choice = prompt("Select an option: ")
        
        if choice == '1':
            create_account(current_os)
            return True
        elif choice == '2':
            return create_account(current_os)
        elif choice == '3':
            create_account(current_os)
        elif choice == '4':
            # Add logic to display user manual
            pass
        elif choice == '0':
            sys.exit(0)
        else:
            error("Invalid option selected")
        
        prompt("Press Enter to continue...")

def main():
    current_os = determine_os()
    
    account = load_account()
    
    if account:
        main_menu_logged_in(account, current_os)
    else:
        if main_menu_logged_out(current_os):
            account = load_account()
            main_menu_logged_in(account, current_os)

if __name__ == "__main__":
    print_banner()
    main()
