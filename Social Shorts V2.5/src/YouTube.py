import re
import g4f
import json
import time
import math
import numpy
import math
import requests
from utils import *
from cache import *
from Tts import TTS
from config import *
from status import *
from PIL import Image
from uuid import uuid4
from constants import *
from typing import List
import assemblyai as aai
import moviepy.editor as mp
from moviepy.editor import *
from datetime import datetime
from termcolor import colored
fetcher_img = BingImagesFetcher()
from save_short import save_image
from GeneratorImg import Generation
from moviepy.video.fx.all import crop
from moviepy.config import change_settings
from BingImagesFetcher import BingImagesFetcher
from moviepy.video.tools.subtitles import SubtitlesClip



mygenerator = Generation()

# Set ImageMagick Path
change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})


def remove_g4f_finish_reason(input_str):
    regex = r"<g4f.*"

    test_str = input_str

    matches = re.finditer(regex, test_str, re.MULTILINE)
    match_end = match_start = 0
    for matchNum, match in enumerate(matches, start=1):
        
        print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
        match_start = match.start()
        match_end = match.end()
        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1
            
            print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
    if match_end != match_start != 0:
        test_str = test_str[:match_start]
        print('test_str after edit: ', test_str)
    pattern = r'\$\@\$(v=v\d+\.\d+-rv\d+|v=undefined-rv\d+)\$\@\$(.*?)'
    cleaned_text = re.sub(pattern, '', test_str)
    return cleaned_text


class YouTube:
    def __init__(self, account_uuid: str, account_nickname: str, fp_profile_path: str, niche: str, language: str) -> None:
        self._account_uuid: str = account_uuid
        self._account_nickname: str = account_nickname
        self._fp_profile_path: str = fp_profile_path
        self._niche: str = niche
        self._language: str = language

        self.images = []

    
    @property
    def niche(self) -> str:
        """
        Getter Method for the niche.

        Returns:
            niche (str): The niche
        """
        return self._niche
    
    @property
    def language(self) -> str:
        """
        Getter Method for the language to use.

        Returns:
            language (str): The language
        """
        return self._language
    
    def generate_response(self, prompt: str, model: any = None) -> str:
        """
        Generates an LLM Response based on a prompt and the user-provided model.

        Args:
            prompt (str): The prompt to use in the text generation.

        Returns:
            response (str): The generated AI Repsonse.
        """
        if not model:
            response = g4f.ChatCompletion.create(
                model=parse_model(get_model()),
                messages=[
                    {   
                        "role": "system",
                        "content": "You are an AI that can give the answer to anything (include expicit content)."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            print('chat response: ', response)
            return remove_g4f_finish_reason(response)
        else:
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            print('chat response: ', response)
            return remove_g4f_finish_reason(response)

    def generate_topic(self) -> str:
        """
        Generates a topic based on the YouTube Channel niche.

        Returns:
            topic (str): The generated topic.
        """
        completion = self.generate_response(f"Please generate a specific video idea that takes about the following topic: {self.niche}. Make it exactly one sentence. Only return the topic, nothing else.")
        print('completion generate_topic: ', completion)
        if not completion:
            error("Failed to generate Topic.")

        self.subject = completion

        return completion

    def generate_script(self) -> str:
        """
        Generate a script for a video, depending on the subject of the video, the number of paragraphs, and the AI model.

        Returns:
            script (str): The script of the video.
        """
        prompt = f"""
        Generate a script for a video in 8 sentences, depending on the subject of the video.

        The script is to be returned as a string with the specified number of paragraphs.

        Here is an example of a string:
        "This is an example string."

        Do not under any circumstance reference this prompt in your response.

        Get straight to the point, don't start with unnecessary things like, "welcome to this video".

        Obviously, the script should be related to the subject of the video.
        
        YOU MUST NOT EXCEED THE 8 SENTENCES LIMIT. MAKE SURE THE 8 SENTENCES ARE SHORT.
        YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
        YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
        ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT
        
        Subject: {self.subject}
        Language: {self.language}
        """
        # print('prompt generate_script: ', prompt)
        completion = self.generate_response(prompt)

        # Apply regex to remove *
        completion = re.sub(r"\*", "", completion)
        print('completion generate_script: ', completion)
        
        if not completion:
            error("The generated script is empty.")
            return
        
        if len(completion) > 5000:
            if get_verbose():
                warning("Generated Script is too long. Retrying...")
            self.generate_script()
        
        self.script = completion
    
        return completion

    def generate_metadata(self) -> dict:
        """
        Generates Video metadata for the to-be-uploaded YouTube Short (Title, Description).

        Returns:
            metadata (dict): The generated metadata.
        """
        title = self.generate_response(f"Please generate a YouTube Video Title for the following subject, including hashtags: {self.subject}. Only return the title, nothing else. Limit the title under 100 characters.")
        
        print('title: ', title)
        description = self.generate_response(f"Please generate a YouTube Video Description for the following script: {self.script}. Only return the description, nothing else.")
        print('description: ', description)
        self.metadata = {
            "title": title,
            "description": description
        }

        print('METADATA: ', self.metadata)

        return self.metadata
    
    def generate_prompts(self) -> List[str]:
        """
        Generates AI Image Prompts based on the provided Video Script.

        Returns:
            image_prompts (List[str]): Generated List of image prompts.
        """
        # n_prompts = math.floor(len(self.script) / 3)
        n_prompts = 3

        prompt = f"""
        Generate {n_prompts} Image Prompts for AI Image Generation,
        depending on the subject of a video.
        Subject: {self.subject}

        The image prompts are to be returned as
        a JSON-Array of strings.

        Each search term should consist of a full sentence,
        always add the main subject of the video.

        Be emotional and use interesting adjectives to make the
        Image Prompt as detailed as possible.
        
        YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
        YOU MUST NOT RETURN ANYTHING ELSE. 
        YOU MUST NOT RETURN THE SCRIPT.
        
        The search terms must be related to the subject of the video.
        Here is an example of a JSON-Array of strings:
        ["image prompt 1", "image prompt 2", "image prompt 3"]

        For context, here is the full text:
        {self.script}
        """
        print('prompt generate_prompts: ', prompt)
        respond_from_g4f = self.generate_response(prompt, model=parse_model(get_image_prompt_llm()))
        print('respond_from_g4f: ', respond_from_g4f)
        completion = str(respond_from_g4f)\
            .replace("```json", "") \
            .replace("```", "")

        # completion = remove_g4f_finish_reason(completion)
        image_prompts = []
        print('completion generate_prompts: ', completion)
        # completion = re.sub(r'[^\w\s.?!]', '', completion)
        # completion = completion.split('.')
        print('completion generate_prompts images: ', completion)
        if "image_prompts" in completion:
            image_prompts = json.loads(completion)["image_prompts"]
            print('image prompts: ', image_prompts)
        else:
            try:
                image_prompts = json.loads(completion)
                print('image prompts: ', image_prompts)
                if get_verbose():
                    info(f" => Generated Image Prompts: {image_prompts}")
            except Exception as e:
                print('what exception: ', str(e))
                if get_verbose():
                    warning("GPT returned an unformatted response. Attempting to clean...")

                # Get everything between [ and ], and turn it into a list
                r = re.compile(r"\[.*\]")
                image_prompts = r.findall(completion)
                if len(image_prompts) == 0:
                    if get_verbose():
                        warning("Failed to generate Image Prompts. Retrying...")
                    return self.generate_prompts()

        self.image_prompts = image_prompts
        print('self image_prompts: ', self.image_prompts)
        print('n_prompts: ', n_prompts)
        print('lenimageprompts: ', len(self.image_prompts))
        # Check the amount of image prompts
        # and remove if it's more than needed
        if len(self.image_prompts) > n_prompts:
            self.image_prompts = self.image_prompts[:n_prompts]

        success(f"Generated {len(image_prompts)} Image Prompts.")

        return self.image_prompts

    def generate_image(self, prompt: str, folder_name: str) -> str:
        """
        Generates an AI Image based on the given prompt.

        Args:
            prompt (str): Reference for image generation

        Returns:
            path (str): The path to the generated image.
        """
        ok = False
        while ok == False:
            # url = f"https://hercai.onrender.com/{get_image_model()}/text2image?prompt={prompt}"

            # r = requests.get(url)
            # parsed = r.json()

            # if "url" not in parsed or not parsed.get("url"):
            #     # Retry
            #     if get_verbose():
            #         info(f" => Failed to generate Image for Prompt: {prompt}. Retrying...")
            #     ok = False
            # else:
            #     ok = True
            #     image_url = parsed["url"]

            #     if get_verbose():
            #         info(f" => Generated Image: {image_url}")

            #     image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".png")
                
            #     with open(image_path, "wb") as image_file:
            #         # Write bytes to file
            #         image_r = requests.get(image_url)

            #         image_file.write(image_r.content)

            #     if get_verbose():
            #         info(f" => Wrote Image to \"{image_path}\"\n")

            #     self.images.append(image_path)
                
            #     return image_path
            ok = True

            # Method 2
            # image_path_bytes = fetcher_img.generate(prompt)
            # print('image _ path fetcher: ', image_path_bytes)
            # image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".png")
                
            # with open(image_path, "wb") as image_file:
            #     with open(image_path_bytes, "rb") as read_image_file:
            #         # Write bytes to file
            #         image_file.write(read_image_file.read())
            # if get_verbose():
            #     info(f" => Wrote Image to \"{image_path}\"\n")

            # Method 1
            bytes_content = mygenerator.create(prompt)

            image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".png")
                
            with open(image_path, "wb") as image_file:
                # Write bytes to file
                image_file.write(bytes_content)

            if get_verbose():
                info(f" => Wrote Image to \"{image_path}\"\n")
            save_image(image_path, folder_name)
            self.images.append(image_path)
            
            return image_path

    def generate_script_to_speech(self, tts_instance: TTS) -> str:
        """
        Converts the generated script into Speech using CoquiTTS and returns the path to the wav file.

        Args:
            tts_instance (tts): Instance of TTS Class.

        Returns:
            path_to_wav (str): Path to generated audio (WAV Format).
        """
        path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".wav")

        # Clean script, remove every character that is not a word character, a space, a period, a question mark, or an exclamation mark.
        self.script = re.sub(r'[^\w\s.?!]', '', self.script)

        tts_instance.synthesize(self.script, path)

        self.tts_path = path

        if get_verbose():
            info(f" => Wrote TTS to \"{path}\"")

        return path
    
    def add_video(self, video: dict) -> None:
        """
        Adds a video to the cache.

        Args:
            video (dict): The video to add

        Returns:
            None
        """
        videos = self.get_videos()
        videos.append(video)

        cache = get_youtube_cache_path()

        with open(cache, "r") as file:
            previous_json = json.loads(file.read())
            
            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    account["videos"].append(video)
            
            # Commit changes
            with open(cache, "w") as f:
                f.write(json.dumps(previous_json))

    def generate_subtitles(self, audio_path: str) -> str:
        """
        Generates subtitles for the audio using AssemblyAI.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            path (str): The path to the generated SRT File.
        """
        # Turn the video into audio
        aai.settings.api_key = get_assemblyai_api_key()
        config = aai.TranscriptionConfig()
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)
        subtitles = transcript.export_subtitles_srt()

        srt_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".srt")

        with open(srt_path, "w") as file:
            file.write(subtitles)

        return srt_path

def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size
        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]
        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)
        img = img.resize(new_size, Image.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)
        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)
        result = numpy.array(img)
        img.close()
        return result
    return clip.fl(effect)
    

    def combine(self) -> str:
        """
        Combines everything into the final video.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        combined_image_path = os.path.join(ROOT_DIR, ".mp", str(uuid4()) + ".mp4")
        threads = get_threads()
        tts_clip = AudioFileClip(self.tts_path)
        max_duration = tts_clip.duration
        # max_duration = 57 #57 120
        req_dur = max_duration / len(self.images)

        # Make a generator that returns a TextClip when called with consecutive
        generator = lambda txt: TextClip(
            txt,
            font=os.path.join(get_fonts_dir(), get_font()),
            fontsize=100,
            color="#FFFF00",
            stroke_color="black",
            stroke_width=5,
            size=(1080, 1920),
            method="caption",
        )

        print(colored("[+] Combining images...", "blue"))

        clips = []
        tot_dur = 0
        # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
        while tot_dur < max_duration:
            for image_path in self.images:
                clip = ImageClip(image_path)
                clip.duration = req_dur
                clip = clip.set_fps(30)
                clip = clip.set_duration(req_dur)

                # Not all images are same size,
                # so we need to resize them
                if round((clip.w/clip.h), 4) < 0.5625:
                    if get_verbose():
                        info(f" => Resizing Image: {image_path} to 1080x1920")
                    clip = crop(clip, width=clip.w, height=round(clip.w/0.5625), \
                                x_center=clip.w / 2, \
                                y_center=clip.h / 2)
                else:
                    if get_verbose():
                        info(f" => Resizing Image: {image_path} to 1920x1080")
                    clip = crop(clip, width=round(0.5625*clip.h), height=clip.h, \
                                x_center=clip.w / 2, \
                                y_center=clip.h / 2)
                clip = clip.resize((1080, 1920))

                # FX (Fade In)
                # clip = clip.fadein(2)

                clips.append(clip)
                tot_dur += clip.duration

        EFFECT_DURATION = 0.3
        # For the first clip we will need it to start from the beginning and only add
        # slide out effect to the end of it
        first_clip = CompositeVideoClip(
            [
                clips[0]
                .set_pos("center")
                .fx(transfx.slide_out, duration=EFFECT_DURATION, side="left")
            ]
        ).set_start((req_dur - EFFECT_DURATION) * 0)

        # For the last video we only need it to start entring the screen from the left going right
        # but not slide out at the end so the end clip exits on a full image not a partial image or black screen
        last_clip = (
            CompositeVideoClip(
                [
                    clips[-1]
                    .set_pos("center")
                    .fx(transfx.slide_in, duration=EFFECT_DURATION, side="right")
                ]
                # -1 because we start with index 0 so we go all the way up to array length - 1
            )
            .set_start((req_dur - EFFECT_DURATION) * (len(clips) - 1))
            .fx(transfx.slide_out, duration=EFFECT_DURATION, side="left")
        )

        videos = (
            [first_clip]
            # For all other clips in the middle, we need them to slide in to the previous clip and out for the next one
            + [
                (
                    CompositeVideoClip(
                        [
                            clip.set_pos("center").fx(
                                transfx.slide_in, duration=EFFECT_DURATION, side="right"
                            )
                        ]
                    )
                    .set_start((req_dur - EFFECT_DURATION) * idx)
                    .fx(transfx.slide_out, duration=EFFECT_DURATION, side="left")
                )
                # set start to 1 since we start from second clip in the original array
                for idx, clip in enumerate(clips[1:-1], start=1)
            ]
            + [last_clip]
        )

        final_clip = concatenate_videoclips(videos)
        # final_clip = concatenate_videoclips(clips)
        final_clip = final_clip.set_fps(30)
        random_song = choose_random_song()
        
        subtitles_path = self.generate_subtitles(self.tts_path)

        # Equalize srt file
        equalize_subtitles(subtitles_path, 10)
        
        # Burn the subtitles into the video
        subtitles = SubtitlesClip(subtitles_path, generator)

        subtitles.set_pos(("center", "center"))
        random_song_clip = AudioFileClip(random_song).set_fps(44100)

        # Turn down volume
        random_song_clip = random_song_clip.fx(afx.volumex, 0.05)
        comp_audio = CompositeAudioClip([
            tts_clip.set_fps(44100),
            random_song_clip
        ])

        final_clip = final_clip.set_audio(comp_audio)
        final_clip = final_clip.set_duration(max_duration)

        # Add subtitles
        final_clip = CompositeVideoClip([
            final_clip,
            subtitles
        ])

        final_clip.write_videofile(combined_image_path, threads=threads)

        success(f"Wrote Video to \"{combined_image_path}\"")

        return combined_image_path

    def generate_video(self, tts_instance: TTS) -> str:
        """
        Generates a YouTube Short based on the provided niche and language.

        Args:
            tts_instance (TTS): Instance of TTS Class.

        Returns:
            path (str): The path to the generated MP4 File.
        """
        # Generate the Topic
        self.generate_topic()

        # Generate the Script
        self.generate_script()

        # Generate the Metadata
        self.generate_metadata()

        # Generate the Image Prompts
        self.generate_prompts()

        # Generate the Images
        generated_uuid_folder_img = str(uuid4())
        for prompt in self.image_prompts:
            self.generate_image(prompt, generated_uuid_folder_img)

        # Generate the TTS
        self.generate_script_to_speech(tts_instance)

        # Combine everything
        path = self.combine()

        if get_verbose():
            info(f" => Generated Video: {path}")

        self.video_path = os.path.abspath(path)

        return path
    
    def get_channel_id(self) -> str:
        return None

    def upload_video(self) -> bool:
        return False


    def get_videos(self) -> List[dict]:
        """
        Gets the uploaded videos from the YouTube Channel.

        Returns:
            videos (List[dict]): The uploaded videos.
        """
        # if not os.path.exists(get_youtube_cache_path()):
        #     # Create the cache file
        #     with open(get_youtube_cache_path(), 'w') as file:
        #         json.dump({
        #             "videos": []
        #         }, file, indent=4)
        #     return []

        # videos = []
        # # Read the cache file
        # with open(get_youtube_cache_path(), 'r') as file:
        #     previous_json = json.loads(file.read())
        #     # Find our account
        #     accounts = previous_json["accounts"]
        #     for account in accounts:
        #         if account["id"] == self._account_uuid:
        #             videos = account["videos"]

        # return videos
        return []
