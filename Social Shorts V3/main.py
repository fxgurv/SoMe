import os
from termcolor import colored
from text import Text
from tts import TTS
from transcribe import Transcribe, APIKeyManager
from image import ImageGenerator
from browse import Browser

def main():
    try:
        api_key_manager = APIKeyManager()
    except ValueError as e:
        print(colored(f"Error: {str(e)}", "green"))
        return

    options = [
        "Text Generation",
        "Text-to-Speech",
        "Speech-to-Text",
        "Text to Image",
        "Web Browsing",
        "Exit"
    ]

    while True:
        print(colored("\nMain Menu:", "green"))
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        choice = input(colored("\nEnter your choice (1-6): ", "yellow"))

        if choice == "6":
            print(colored("Exiting...", "green"))
            break

        if choice == "1":
            text = Text()
            text_options = [
                "Generate Response using Gemini",
                "Generate Response Using OpenAI",
                "Generate Response using Claude AI",
                "Generate Response using Local AI",
                "Generate Response Using G4F Models",
                "Back to Main Menu"
            ]
            while True:
                print(colored("\nText Generation Options:", "cyan"))
                for i, option in enumerate(text_options, 1):
                    print(f"{i}. {option}")

                text_choice = input(colored("\nEnter your choice (1-6): ", "yellow"))

                if text_choice == "6":
                    break

                if text_choice in ["1", "2", "3", "4", "5"]:
                    prompt = input(colored(f"\nWrite message to test {text_options[int(text_choice)-1]}: ", "yellow"))
                    
                    if text_choice == "1":
                        response = text.gemini(prompt)
                    elif text_choice == "2":
                        response = text.openai(prompt)
                    elif text_choice == "3":
                        response = text.claude(prompt)
                    elif text_choice == "4":
                        response = text.local_ai(prompt)
                    elif text_choice == "5":
                        response = text.g4f(prompt)

                    print(colored(f"\n{text_options[int(text_choice)-1]}: {response}", "green"))
                else:
                    print(colored("Invalid choice. Please try again.", "red"))

        elif choice == "2":
            tts = TTS()
            tts_options = [
                "Test Google TTS",
                "Test Edge TTS",
                "Test OpenAI TTS",
                "Test Elevenlabs TTS",
                "Test Coqui TTS",
                "Back to Main Menu"
            ]
            while True:
                print(colored("\nText-to-Speech Options:", "cyan"))
                for i, option in enumerate(tts_options, 1):
                    print(f"{i}. {option}")

                tts_choice = input(colored("\nEnter your choice (1-6): ", "yellow"))

                if tts_choice == "6":
                    break

                if tts_choice in ["1", "2", "3", "4", "5"]:
                    text = input(colored(f"\nEnter text for {tts_options[int(tts_choice)-1]}: ", "yellow"))
                    
                    if tts_choice == "1":
                        output_file = tts.google_tts(text)
                    elif tts_choice == "2":
                        import asyncio
                        output_file = asyncio.run(tts.edge_tts(text))
                    elif tts_choice == "3":
                        output_file = tts.openai_tts(text)
                    elif tts_choice == "4":
                        output_file = tts.elevenlabs_tts(text)
                    elif tts_choice == "5":
                        output_file = tts.coqui_tts(text)

                    print(colored(f"\nAudio generated and saved as {output_file}", "green"))
                else:
                    print(colored("Invalid choice. Please try again.", "red"))

        elif choice == "3":
            transcribe = Transcribe()
            transcribe_options = [
                "Transcribe using CMU Sphinx",
                "Transcribe using Google Speech-to-Text",
                "Transcribe using Faster Whisper",
                "Transcribe using Vosk",
                "Transcribe using AssemblyAI",
                "Transcribe using Deepgram",
                "Transcribe using Wav2Vec2",
                "Transcribe using NVIDIA NeMo",
                "Back to Main Menu"
            ]
            while True:
                print(colored("\nTranscription Options:", "cyan"))
                for i, option in enumerate(transcribe_options, 1):
                    print(f"{i}. {option}")

                transcribe_choice = input(colored("\nEnter your choice (1-9): ", "yellow"))

                if transcribe_choice == "9":
                    break

                if transcribe_choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                    audio_file = input(colored("\nEnter the path to the audio file: ", "yellow"))
                    
                    if not os.path.exists(audio_file):
                        print(colored("Audio file not found. Please try again.", "red"))
                        continue

                    try:
                        if transcribe_choice == "1":
                            result = transcribe.transcribe_sphinx(audio_file)
                        elif transcribe_choice == "2":
                            result = transcribe.transcribe_google(audio_file)
                        elif transcribe_choice == "3":
                            result = transcribe.transcribe_faster_whisper(audio_file)
                        elif transcribe_choice == "4":
                            result = transcribe.transcribe_vosk(audio_file)
                        elif transcribe_choice == "5":
                            result = transcribe.transcribe_assembly_ai(audio_file)
                        elif transcribe_choice == "6":
                            result = transcribe.transcribe_deepgram(audio_file)
                        elif transcribe_choice == "7":
                            result = transcribe.transcribe_wav2vec(audio_file)
                        elif transcribe_choice == "8":
                            result = transcribe.transcribe_nemo(audio_file)

                        print(colored(f"\nTranscription result: {result}", "green"))
                    except Exception as e:
                        print(colored(f"Error during transcription: {str(e)}", "red"))
                else:
                    print(colored("Invalid choice. Please try again.", "red"))

        elif choice == "4":
            image_gen = ImageGenerator()
            image_options = [
                "Generate image using DALL-E",
                "Generate image using Stable Diffusion",
                "Generate images using Segmind",
                "Generate images using Hercai",
                "Generate images using Prodia",
                "Generate images using Pollinations",
                "Generate images using DeepAI",
                "Back to Main Menu"
            ]
            while True:
                print(colored("\nImage Generation Options:", "cyan"))
                for i, option in enumerate(image_options, 1):
                    print(f"{i}. {option}")

                image_choice = input(colored("\nEnter your choice (1-8): ", "yellow"))

                if image_choice == "8":
                    break

                if image_choice in ["1", "2", "3", "4", "5", "6", "7"]:
                    prompt = input(colored("\nEnter the image prompt: ", "yellow"))
                    
                    if image_choice == "1":
                        output_file = image_gen.generate_dalle(prompt)
                        print(colored(f"\nImage generated and saved as {output_file}", "green"))
                    elif image_choice == "2":
                        output_file = image_gen.generate_stable_diffusion(prompt)
                        print(colored(f"\nImage generated and saved as {output_file}", "green"))
                    else:
                        num_images = int(input(colored("\nEnter the number of images to generate: ", "yellow")))
                        prompts = [prompt] * num_images
                        output_folder = f"{image_options[int(image_choice)-1].split()[-1].lower()}_output"
                        
                        if image_choice == "3":
                            image_gen.segmind(prompts, output_folder)
                        elif image_choice == "4":
                            image_gen.hercai(prompts, output_folder)
                        elif image_choice == "5":
                            image_gen.prodia(prompts, output_folder)
                        elif image_choice == "6":
                            image_gen.pollinations(prompts, output_folder)
                        elif image_choice == "7":
                            image_gen.deepai(prompts, output_folder)
                        
                        print(colored(f"\nImages generated and saved in folder: {output_folder}", "green"))
                else:
                    print(colored("Invalid choice. Please try again.", "red"))

        elif choice == "5":
            browser = Browser()
            browser_options = [
                "Open Chrome",
                "Open Firefox",
                "Open Safari",
                "Open Edge",
                "Back to Main Menu"
            ]
            while True:
                print(colored("\nBrowser Options:", "cyan"))
                for i, option in enumerate(browser_options, 1):
                    print(f"{i}. {option}")

                browser_choice = input(colored("\nEnter your choice (1-5): ", "yellow"))

                if browser_choice == "5":
                    break

                if browser_choice in ["1", "2", "3", "4"]:
                    headless = input(colored("Run in headless mode? (y/n): ", "yellow")).lower() == 'y'
                    
                    if browser_choice == "1":
                        driver = browser.get_browser('chrome', headless)
                    elif browser_choice == "2":
                        driver = browser.get_browser('firefox', headless)
                    elif browser_choice == "3":
                        driver = browser.get_browser('safari', headless)
                    elif browser_choice == "4":
                        driver = browser.get_browser('edge', headless)

                    url = input(colored("Enter the URL to navigate to: ", "yellow"))
                    driver.get(url)
                    print(colored(f"Navigated to {url}", "green"))
                    
                    input(colored("Press Enter to close the browser...", "yellow"))
                    browser.close_browser()
                else:
                    print(colored("Invalid choice. Please try again.", "red"))

        else:
            print(colored("Invalid choice. Please try again.", "red"))

if __name__ == "__main__":
    main()
