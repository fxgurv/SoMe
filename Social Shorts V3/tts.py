import os
import sys
import io
import warnings
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
import requests
from termcolor import colored
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTS:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        self.local_ai_url = os.getenv('LOCAL_AI_URL', 'https://imseldrith-tts-openai-free.hf.space/v1/audio/speech')
        self.tts_voice = "alloy"  # Default voice

    def google_tts(self, text, output_file="google_tts_output.mp3"):
        from gtts import gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        return output_file

    async def edge_tts(self, text, output_file="edge_tts_output.mp3"):
        import edge_tts
        communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
        await communicate.save(output_file)
        return output_file

    def openai_tts(self, text, output_file="openai_tts_output.mp3"):
        import requests
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": "alloy"
        }
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return output_file
        return None

    def local_ai_tts(self, text, output_file="local_ai_tts_output.mp3"):
        logger.info("Synthesizing text using Local AI TTS")
        headers = {
            "accept": "*/*",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": self.tts_voice,
            "response_format": "mp3",
            "speed": 1
        }
        try:
            response = requests.post(
                f"{self.local_ai_url}/v1/audio/speech",
                json=data,
                headers=headers
            )
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

            logger.info(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except requests.exceptions.RequestException as e:
            logger.error(f"Error with Local AI TTS: {str(e)}")
            return None

    def elevenlabs_tts(self, text, output_file="elevenlabs_tts_output.mp3"):
        import requests
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return output_file
        return None

    def coqui_tts(self, text, output_file="coqui_tts_output.wav"):
        from TTS.api import TTS as CoquiTTS
        tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
        tts.tts_to_file(text=text, file_path=output_file)
        return output_file

    def mozilla_tts(self, text, output_file="mozilla_tts_output.wav"):
        from TTS.api import TTS as MozillaTTS
        tts = MozillaTTS("tts_models/en/ljspeech/mozilla-tts")
        tts.tts_to_file(text=text, file_path=output_file)
        return output_file

    def fairseq_tts(self, text, output_file="fairseq_tts_output.wav"):
        import torch
        import soundfile as sf
        from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
        
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech"
        )
        model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        generator = task.build_generator([model], cfg)
        
        sample = TTSHubInterface.get_model_input(task, text)
        wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
        sf.write(output_file, wav, rate)
        return output_file

def main():
    tts = TTS()
    options = [
        "Test Google TTS",
        "Test Edge TTS",
        "Test OpenAI TTS",
        "Test Local AI TTS",
        "Test Elevenlabs TTS",
        "Test Coqui TTS",
        "Test Mozilla TTS",
        "Test Fairseq TTS",
        "Exit"
    ]

    while True:
        print(colored("\nText-to-Speech Options:", "cyan"))
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        choice = input(colored("\nEnter your choice (1-9): ", "yellow"))

        if choice == "9":
            print(colored("Exiting...", "green"))
            break

        if choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            text = input(colored(f"\nEnter text for {options[int(choice)-1]}: ", "yellow"))
            
            try:
                if choice == "1":
                    output_file = tts.google_tts(text)
                elif choice == "2":
                    output_file = asyncio.run(tts.edge_tts(text))
                elif choice == "3":
                    output_file = tts.openai_tts(text)
                elif choice == "4":
                    output_file = tts.local_ai_tts(text)
                elif choice == "5":
                    output_file = tts.elevenlabs_tts(text)
                elif choice == "6":
                    output_file = tts.coqui_tts(text)
                elif choice == "7":
                    output_file = tts.mozilla_tts(text)
                elif choice == "8":
                    output_file = tts.fairseq_tts(text)

                if output_file:
                    print(colored(f"\nAudio generated and saved as {output_file}", "green"))
                else:
                    print(colored("Failed to generate audio", "red"))
            except Exception as e:
                print(colored(f"Error: {str(e)}", "red"))
        else:
            print(colored("Invalid choice. Please try again.", "red"))

if __name__ == "__main__":
    main()
