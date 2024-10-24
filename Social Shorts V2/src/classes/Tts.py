import os
import sys
import io
import warnings
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
from status import info, warning, error, success
from config import ROOT_DIR, get_tts_engine, get_tts_voice, get_elevenlabs_api_key, get_openai_api_key
import requests
from gtts import gTTS

class TTS:
    def __init__(self) -> None:
        info("Initializing TTS class")
        self.tts_engine = get_tts_engine()
        self.tts_voice = get_tts_voice()
        self.elevenlabs_api_key = get_elevenlabs_api_key() if self.tts_engine == 'elevenlabs' else None
        self.openai_api_key = get_openai_api_key() if self.tts_engine == 'openai' else None
        info(f"TTS Engine: {self.tts_engine}, Voice: {self.tts_voice}")

        if self.tts_engine == 'coqui':
            self._init_coqui_tts()
        elif self.tts_engine == 'edge':
            self._init_edge_tts()

    def _init_coqui_tts(self):
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        site_packages = next((p for p in sys.path if 'site-packages' in p), None)
        if not site_packages:
            raise EnvironmentError("Could not find site-packages directory. Ensure that the TTS package is installed.")

        models_json_path = os.path.join(site_packages, "TTS", ".models.json")
        self._model_manager = ModelManager(models_json_path)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self._model_path, self._config_path, self._model_item = \
                    self._model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC_ph")

                voc_path, voc_config_path, _ = self._model_manager. \
                    download_model("vocoder_models/en/ljspeech/univnet")
                
                self._synthesizer = Synthesizer(
                    tts_checkpoint=self._model_path,
                    tts_config_path=self._config_path,
                    vocoder_checkpoint=voc_path,
                    vocoder_config=voc_config_path
                )

    def _init_edge_tts(self):
        import edge_tts
        self.edge_tts = edge_tts

    def synthesize(self, text: str, output_file: str = os.path.join(ROOT_DIR, "tmp", "audio.wav")) -> str:
        info(f"Synthesizing text using {self.tts_engine}")
        if self.tts_engine == 'elevenlabs':
            return self._synthesize_elevenlabs(text, output_file)
        elif self.tts_engine == 'openai':
            return self._synthesize_openai(text, output_file)
        elif self.tts_engine == 'gtts':
            return self._synthesize_gtts(text, output_file)
        elif self.tts_engine == 'coqui':
            return self._synthesize_coqui(text, output_file)
        elif self.tts_engine == 'edge':
            return self._synthesize_edge(text, output_file)
        elif self.tts_engine == 'local_openai':
            return self._synthesize_local_openai(text, output_file)
        else:
            error(f"Unsupported TTS engine: {self.tts_engine}")
            raise ValueError(f"Unsupported TTS engine: {self.tts_engine}")

    def _synthesize_elevenlabs(self, text: str, output_file: str) -> str:
        info("Synthesizing text using ElevenLabs")
        
        if not self.elevenlabs_api_key:
            raise ValueError("ElevenLabs API key is not set")

        voices_url = "https://api.elevenlabs.io/v1/voices"
        headers = {
            "xi-api-key": self.elevenlabs_api_key
        }

        try:
            voices_response = requests.get(voices_url, headers=headers)
            voices_response.raise_for_status()
            voices = voices_response.json().get("voices", [])
            
            voice_id = next((voice["voice_id"] for voice in voices if voice["name"].lower() == self.tts_voice.lower()), None)
            
            if not voice_id:
                raise ValueError(f"Voice '{self.tts_voice}' not found")

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers.update({
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            })
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

            success(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except requests.exceptions.HTTPError as http_err:
            error(f"HTTP error occurred: {http_err}")
            if response.status_code == 401:
                error("ElevenLabs API error: Unauthorized. Please check your API key.")
            elif response.status_code == 400:
                error_message = response.json().get('detail', {}).get('message', 'Unknown error')
                error(f"ElevenLabs API error: {error_message}")
            raise
        except Exception as err:
            error(f"An error occurred: {err}")
            raise

    def _synthesize_openai(self, text: str, output_file: str) -> str:
        info("Synthesizing text using OpenAI")
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": self.tts_voice
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

            success(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except requests.exceptions.HTTPError as http_err:
            error(f"HTTP error occurred: {http_err}")
            if response.status_code == 400:
                error_message = response.json().get('error', {}).get('message', 'Unknown error')
                error(f"OpenAI API error: {error_message}")
            raise
        except Exception as err:
            error(f"An error occurred: {err}")
            raise

    def _synthesize_gtts(self, text: str, output_file: str) -> str:
        info("Synthesizing text using gTTS")
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_file)
            success(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except Exception as err:
            error(f"An error occurred with gTTS: {err}")
            raise

    def _synthesize_coqui(self, text: str, output_file: str) -> str:
        info("Synthesizing text using Coqui TTS")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    outputs = self._synthesizer.tts(text)
                    self._synthesizer.save_wav(outputs, output_file)
            success(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except Exception as err:
            error(f"An error occurred with Coqui TTS: {err}")
            raise

    def _synthesize_edge(self, text: str, output_file: str) -> str:
        info("Synthesizing text using Edge TTS")
        try:
            communicate = self.edge_tts.Communicate(text, self.tts_voice)
            async def _main():
                await communicate.save(output_file)
            import asyncio
            asyncio.run(_main())
            success(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except Exception as err:
            error(f"An error occurred with Edge TTS: {err}")
            raise

    def _synthesize_local_openai(self, text: str, output_file: str) -> str:
        info("Synthesizing text using Local OpenAI TTS")
        url = "https://imseldrith-tts-openai-free.hf.space/v1/audio/speech"
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
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

            success(f"Audio synthesized successfully and saved to {output_file}")
            return output_file
        except requests.exceptions.HTTPError as http_err:
            error(f"HTTP error occurred: {http_err}")
            error_message = response.json().get('error', 'Unknown error')
            error(f"Local OpenAI TTS API error: {error_message}")
            raise
        except Exception as err:
            error(f"An error occurred: {err}")
            raise
