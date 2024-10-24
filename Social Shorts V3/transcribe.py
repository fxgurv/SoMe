import os
import json
import requests
from typing import Optional
from termcolor import colored
import configparser
import toml

class APIKeyManager:
    def __init__(self):
        self.api_keys = {}
        self.load_api_keys()

    def load_api_keys(self):
        # Check config.env
        if os.path.exists('config.env'):
            self.read_file('config.env')
        
        # Check config.toml
        if os.path.exists('config.toml'):
            try:
                with open('config.toml', 'r', encoding='utf-8') as f:
                    toml_data = toml.load(f)
                    self.api_keys.update(toml_data.get('api_keys', {}))
            except Exception as e:
                print(f"Error reading config.toml: {e}")
        
        # Check config.py
        try:
            import config
            self.api_keys.update({k: v for k, v in config.__dict__.items() if not k.startswith('__')})
        except ImportError:
            pass
        
        # Check .ini files
        for file in os.listdir():
            if file.endswith('.ini'):
                config = configparser.ConfigParser()
                try:
                    config.read(file, encoding='utf-8')
                    if 'API_KEYS' in config:
                        self.api_keys.update(dict(config['API_KEYS']))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        # Check environment variables
        for key in ['ASSEMBLY_API_KEY', 'DEEPGRAM_API_KEY', 'REV_AI_KEY']:
            if key in os.environ:
                self.api_keys[key] = os.environ[key]
        
        # Check all other files
        for file in os.listdir():
            if os.path.isfile(file) and file not in ['config.env', 'config.toml', 'config.py']:
                self.read_file(file)

    def read_file(self, file):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        if key in ['ASSEMBLY_API_KEY', 'DEEPGRAM_API_KEY', 'REV_AI_KEY'] and key not in self.api_keys:
                            self.api_keys[key] = value
        except UnicodeDecodeError:
            try:
                with open(file, 'r', encoding='latin-1') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key in ['ASSEMBLY_API_KEY', 'DEEPGRAM_API_KEY', 'REV_AI_KEY'] and key not in self.api_keys:
                                self.api_keys[key] = value
            except Exception as e:
                print(f"Error reading {file}: {e}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    def get_api_key(self, key):
        if key not in self.api_keys:
            raise ValueError(f"API key '{key}' not found in any configuration file or environment variable.")
        return self.api_keys[key]

class Transcribe:
    def __init__(self):
        self.api_key_manager = APIKeyManager()
        self.assembly_ai_key = self.api_key_manager.get_api_key('ASSEMBLY_API_KEY')
        self.deepgram_api_key = self.api_key_manager.get_api_key('DEEPGRAM_API_KEY')
        self.rev_ai_key = self.api_key_manager.get_api_key('REV_AI_KEY')

    def transcribe_sphinx(self, audio_file):
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return "Sphinx could not understand audio"
        except sr.RequestError as e:
            return f"Sphinx error; {e}"

    def transcribe_google(self, audio_file):
        from google.cloud import speech
        client = speech.SpeechClient()
        with open(audio_file, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        response = client.recognize(config=config, audio=audio)
        return " ".join(result.alternatives[0].transcript for result in response.results)

    def transcribe_faster_whisper(self, audio_file):
        from faster_whisper import WhisperModel
        model_size = "base"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_file, beam_size=5)
        return " ".join(segment.text for segment in segments)

    def transcribe_vosk(self, audio_file):
        from vosk import Model, KaldiRecognizer
        import wave
        
        model = Model(lang="en-us")
        wf = wave.open(audio_file, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result.get('text', ''))
        
        part_result = json.loads(rec.FinalResult())
        results.append(part_result.get('text', ''))
        return " ".join(results)

    def transcribe_assembly_ai(self, audio_file):
        headers = {'authorization': self.assembly_ai_key}
        
        with open(audio_file, 'rb') as f:
            response = requests.post('https://api.assemblyai.com/v2/upload',
                                  headers=headers,
                                  files={'file': f})
            upload_url = response.json()['upload_url']

        json_data = {
            'audio_url': upload_url,
            'language_code': 'en'
        }
        response = requests.post('https://api.assemblyai.com/v2/transcript',
                               json=json_data,
                               headers=headers)
        transcript_id = response.json()['id']

        while True:
            response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                                 headers=headers)
            result = response.json()
            if result['status'] == 'completed':
                return result['text']
            elif result['status'] == 'error':
                return f"AssemblyAI error: {result['error']}"

    def transcribe_deepgram(self, audio_file):
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {self.deepgram_api_key}",
        }
        
        with open(audio_file, 'rb') as f:
            response = requests.post(url,
                                  headers=headers,
                                  data=f,
                                  params={"model": "general", "language": "en-US"})
            
        if response.status_code == 200:
            result = response.json()
            return result['results']['channels'][0]['alternatives'][0]['transcript']
        return f"Deepgram error: {response.text}"

    def transcribe_wav2vec(self, audio_file):
        import torch
        import torchaudio
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)
        
        input_values = processor(waveform.squeeze().numpy(), 
                               sampling_rate=16000, 
                               return_tensors="pt").input_values
        
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription

    def transcribe_nemo(self, audio_file):
        import nemo.collections.asr as nemo_asr
        
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
            model_name="QuartzNet15x5Base-En"
        )
        transcription = asr_model.transcribe([audio_file])[0]
        return transcription

def main():
    try:
        transcribe = Transcribe()
        options = [
            "Transcribe using CMU Sphinx",
            "Transcribe using Google Speech-to-Text",
            "Transcribe using Faster Whisper",
            "Transcribe using Vosk",
            "Transcribe using AssemblyAI",
            "Transcribe using Deepgram",
            "Transcribe using Wav2Vec2",
            "Transcribe using NVIDIA NeMo",
            "Exit"
        ]

        while True:
            print(colored("\nTranscription Options:", "cyan"))
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")

            choice = input(colored("\nEnter your choice (1-9): ", "yellow"))

            if choice == "9":
                print(colored("Exiting...", "green"))
                break

            if choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                audio_file = input(colored("\nEnter the path to the audio file: ", "yellow"))
                
                if not os.path.exists(audio_file):
                    print(colored("Audio file not found. Please try again.", "red"))
                    continue

                try:
                    if choice == "1":
                        result = transcribe.transcribe_sphinx(audio_file)
                    elif choice == "2":
                        result = transcribe.transcribe_google(audio_file)
                    elif choice == "3":
                        result = transcribe.transcribe_faster_whisper(audio_file)
                    elif choice == "4":
                        result = transcribe.transcribe_vosk(audio_file)
                    elif choice == "5":
                        result = transcribe.transcribe_assembly_ai(audio_file)
                    elif choice == "6":
                        result = transcribe.transcribe_deepgram(audio_file)
                    elif choice == "7":
                        result = transcribe.transcribe_wav2vec(audio_file)
                    elif choice == "8":
                        result = transcribe.transcribe_nemo(audio_file)

                    print(colored(f"\nTranscription result: {result}", "green"))
                except Exception as e:
                    print(colored(f"Error during transcription: {str(e)}", "red"))
            else:
                print(colored("Invalid choice. Please try again.", "red"))
    except Exception as e:
        print(colored(f"Error: {str(e)}", "red"))

if __name__ == "__main__":
    main()
