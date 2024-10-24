import json
import re
import g4f
import time
import requests
from termcolor import colored
from typing import List, Any
from functools import wraps
from typing import List, Dict, Any
from PIL import Image
import io
import os
import uuid
import random
import torch
from diffusers import StableDiffusionPipeline

# Constants
LOGGING = True
LANGUAGE = "English"
LLM = "gpt4"  # Options: "Gemini", "OpenAI", "Claude"
IMAGE_PROMPT_LLM = "gpt_4o"  # Options: "Gemini", "OpenAI", "Claude"
SEGMIND_API_KEY = "SG_2d3504ba72dbeacc"
IMAGE_SOURCE = "Segmind"  # Options: "Segmind", "Hercai", "StableDiffusion", "Prodia", "Pollinations", "DeepAI"
IMAGE_MODEL = "v3"
GEMINI_API_KEY = "AIzaSyC6N1MVe9WmAFjWMNuXjlaLnYa8eO813tY"
OPENAI_API_KEY = "sk-66QxChCWt342Vhap9ucm-OU1YZrGAUDWZux"
CLAUDE_API_KEY = "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"
PRODIA_API_KEY = "your_prodia_api_key_here"
DEEPAI_API_KEY = "your_deepai_api_key_here"
MAX_RETRIES = 25

# Logging functions
def log_message(message: str, color: str, emoji: str = "") -> None:
    if LOGGING:
        print(colored(f"{emoji} {message}", color))

def error(message: str): log_message(message, "light_red", "❌")
def success(message: str): log_message(message, "light_green", "✅")
def info(message: str): log_message(message, "light_cyan", "ℹ️")
def warning(message: str): log_message(message, "light_yellow", "⚠️")
def question(message: str) -> str: return input(colored(f"❓ {message}", "light_magenta"))

# Retry decorator
def retry(max_attempts: int = MAX_RETRIES):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        error(f"Max retries ({max_attempts}) reached. Last error: {str(e)}")
                        raise
                    warning(f"Attempt {attempt} failed. Retrying... Error: {str(e)}")
                    time.sleep(1)  # Add a small delay between retries
        return wrapper
    return decorator

# Model parsing function
def parse_model(model_name: str) -> any:
    if model_name == "gpt4":
        return g4f.models.gpt_4
    elif model_name == "gpt_4o":
        return g4f.models.gpt_4o
    elif model_name == "gigachat":
        return g4f.models.gigachat
    elif model_name == "meta":
        return g4f.models.meta
    elif model_name == "llama3_8b_instruct":
        return g4f.models.llama3_8b_instruct
    elif model_name == "llama3_70b_instruct":
        return g4f.models.llama3_70b_instruct
    elif model_name == "codellama_34b_instruct":
        return g4f.models.codellama_34b_instruct
    elif model_name == "codellama_70b_instruct":
        return g4f.models.codellama_70b_instruct
    elif model_name == "mixtral_8x7b":
        return g4f.models.mixtral_8x7b
    elif model_name == "mistral_7b":
        return g4f.models.mistral_7b
    elif model_name == "mistral_7b_v02":
        return g4f.models.mistral_7b_v02
    elif model_name == "claude_v2":
        return g4f.models.claude_v2
    elif model_name == "claude_3_opus":
        return g4f.models.claude_3_opus
    elif model_name == "claude_3_sonnet":
        return g4f.models.claude_3_sonnet
    elif model_name == "claude_3_haiku":
        return g4f.models.claude_3_haiku
    elif model_name == "pi":
        return g4f.models.pi
    elif model_name == "dbrx_instruct":
        return g4f.models.dbrx_instruct
    elif model_name == "command_r_plus":
        return g4f.models.command_r_plus
    elif model_name == "blackbox":
        return g4f.models.blackbox
    elif model_name == "reka_core":
        return g4f.models.reka_core
    elif model_name == "nemotron_4_340b_instruct":
        return g4f.models.nemotron_4_340b_instruct
    elif model_name == "Phi_3_mini_4k_instruct":
        return g4f.models.Phi_3_mini_4k_instruct
    elif model_name == "Yi_1_5_34B_Chat":
        return g4f.models.Yi_1_5_34B_Chat
    elif model_name == "Nous_Hermes_2_Mixtral_8x7B_DPO":
        return g4f.models.Nous_Hermes_2_Mixtral_8x7B_DPO
    elif model_name == "llama_2_70b_chat":
        return g4f.models.llama_2_70b_chat
    elif model_name == "gemma_2_9b_it":
        return g4f.models.gemma_2_9b_it
    elif model_name == "gemma_2_27b_it":
        return g4f.models.gemma_2_27b_it
    else:
        return g4f.models.gpt_35_turbo

@retry()
def generate_response(prompt: str, model: Any = None) -> str:
    info(f"Generating response for prompt: {prompt[:50]}...")
    
    if LLM == "Gemini":
        info("Using Google's Gemini model")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response: str = model.generate_content(prompt).text
    elif LLM == "OpenAI":
        info("Using OpenAI model")
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150
        ).choices[0].text.strip()
    elif LLM == "Claude":
        info("Using Claude model")
        import anthropic
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.completions.create(
            model="claude-2",
            prompt=prompt,
            max_tokens_to_sample=300
        ).completion
    else:
        info(f"Using model: {parse_model(LLM) if not model else model}")
        response = g4f.ChatCompletion.create(
            model=parse_model(LLM) if not model else model,
            messages=[{"role": "user", "content": prompt}]
        )
    
    success(f"Response generated, length: {len(response)} characters")
    return response

@retry()
def generate_topic(niche: str) -> str:
    info(f"Generating topic for YouTube video about: {niche}")
    completion = generate_response(f"Please generate a specific video idea that takes about the following topic: {niche}. Make it exactly one sentence. Only return the topic, nothing else.")

    if not completion:
        raise ValueError("Failed to generate Topic.")
    
    success(f"Generated topic: {completion}")
    info(f"Topic character count: {len(completion)}")
    return completion

@retry()
def generate_script(subject: str, language: str) -> str:
    info("Generating script for YouTube video")
    prompt = f"""
    Generate a script for a video in 4 sentences, depending on the subject of the video.
    The script is to be returned as a string with the specified number of paragraphs.
    Here is an example of a string:
    "This is an example string."
    Do not under any circumstance reference this prompt in your response.
    Get straight to the point, don't start with unnecessary things like, "welcome to this video".
    Obviously, the script should be related to the subject of the video.
    
    YOU MUST NOT EXCEED THE 4 SENTENCES LIMIT. MAKE SURE THE 4 SENTENCES ARE SHORT.
    YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
    YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
    ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT
    
    Subject: {subject}
    Language: {language}
    """
    completion = generate_response(prompt)
    completion = re.sub(r"\*", "", completion)
    
    if not completion or len(completion) > 5000:
        raise ValueError("Generated script is empty or too long.")
    
    success(f"Generated script: {completion[:100]}...")
    info(f"Script character count: {len(completion)}")
    return completion

@retry()
def generate_metadata(subject: str, script: str) -> dict:
    info("Generating metadata for YouTube video")
    title = generate_response(f"Please generate a YouTube Video Title for the following subject, including hashtags: {subject}. Only return the title, nothing else. Limit the title under 100 characters.")

    if len(title) > 100:
        raise ValueError("Generated Title is too long.")

    description = generate_response(f"Please generate a YouTube Video Description for the following script: {script}. Only return the description, nothing else.")
    
    metadata = {
        "title": title,
        "description": description
    }
    success(f"Generated metadata:")
    info(f"Title: {title}")
    info(f"Title character count: {len(title)}")
    info(f"Description: {description}")
    info(f"Description character count: {len(description)}")
    return metadata

@retry()
def generate_prompts(script: str, subject: str) -> List[str]:
    info("Generating image prompts for YouTube video")
    n_prompts = 7
    info(f"Number of prompts requested: {n_prompts}")

    prompt = f"""
    Generate {n_prompts} Image Prompts for AI Image Generation,
    depending on the subject of a video.
    Subject: {subject}

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
    {script}
    """
    completion = str(generate_response(prompt, model=parse_model(IMAGE_PROMPT_LLM)))

    try:
        image_prompts = json.loads(completion)
    except json.JSONDecodeError:
        warning("GPT returned an unformatted response. Attempting to clean...")
        r = re.compile(r"\[.*\]")
        image_prompts_match = r.findall(completion)
        if not image_prompts_match:
            raise ValueError("Failed to generate Image Prompts.")
        image_prompts = json.loads(image_prompts_match[0])

    image_prompts = image_prompts[:n_prompts]

    success(f"Generated {len(image_prompts)} Image Prompts.")
    for i, prompt in enumerate(image_prompts, 1):
        info(f"Prompt {i}: {prompt}")
        info(f"Prompt {i} character count: {len(prompt)}")

    return image_prompts

def generate_images_segmind(prompts, fname):
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
    headers = {'x-api-key': SEGMIND_API_KEY}

    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    requests_made = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        if requests_made >= 5 and time.time() - start_time <= 60:
            time_to_wait = 60 - (time.time() - start_time)
            print(f"Waiting for {time_to_wait:.2f} seconds to comply with rate limit...")
            time.sleep(time_to_wait)
            requests_made = 0
            start_time = time.time()

        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        data = {
            "prompt": final_prompt,
            "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
            "style": "hdr",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 30,
            "guidance_scale": 8,
            "strength": 1,
            "seed": random.randint(1, 1000000),
            "img_width": 1024,
            "img_height": 1024,
            "refiner": "yes",
            "base64": False
        }

        response = requests.post(url, json=data, headers=headers)
        requests_made += 1

        if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
            image_data = response.content

            if image_data.startswith(b'\xff\xd8'):
                image = Image.open(io.BytesIO(image_data))
                image_filename = os.path.join(fname, f"{i + 1}.jpg")
                image.save(image_filename)
                print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
            else:
                print("Error: Invalid image data received.")
        else:
            print(f"Error: Failed to generate image. Status code: {response.status_code}")
            print(response.text)

def generate_images_hercai(prompts, fname):
    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)

    currentseed = random.randint(1, 1000000)
    print("seed", currentseed)

    for i, prompt in enumerate(prompts):
        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        url = f"https://hercai.onrender.com/{IMAGE_MODEL}/text2image?prompt={final_prompt}"
        response = requests.get(url)
        if response.status_code == 200:
            parsed = response.json()
            if "url" in parsed and parsed["url"]:
                image_url = parsed["url"]
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image_filename = os.path.join(fname, f"{i + 1}.png")
                    image.save(image_filename)
                    print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                else:
                    print(f"Error: Failed to retrieve image from URL for image {i + 1}")
            else:
                print(f"Error: No image URL in response for image {i + 1}")
        else:
            print(response.text)
            print(f"Error: Failed to retrieve or save image {i + 1}")


def generate_images_prodia(prompts, fname):
    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    headers = {
        "X-Prodia-Key": PRODIA_API_KEY
    }

    for i, prompt in enumerate(prompts):
        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        
        data = {
            "prompt": final_prompt,
            "model": "sdxl",
            "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy",
            "steps": 30,
            "cfg_scale": 7,
            "seed": random.randint(1, 1000000),
            "upscale": True,
            "sampler": "DPM++ 2M Karras"
        }

        response = requests.post("https://api.prodia.com/v1/job", headers=headers, json=data)
        
        if response.status_code == 201:
            job = response.json()
            job_id = job["job"]
            
            while True:
                status_response = requests.get(f"https://api.prodia.com/v1/job/{job_id}", headers=headers)
                status = status_response.json()
                
                if status["status"] == "succeeded":
                    image_url = status["imageUrl"]
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image_data = image_response.content
                        image = Image.open(io.BytesIO(image_data))
                        image_filename = os.path.join(fname, f"{i + 1}.png")
                        image.save(image_filename)
                        print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                        break
                elif status["status"] == "failed":
                    print(f"Error: Job failed for image {i + 1}")
                    break
                
                time.sleep(1)
        else:
            print(f"Error: Failed to start job for image {i + 1}")

def generate_images_pollinations(prompts, fname):
    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)

    for i, prompt in enumerate(prompts):
        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        
        url = f"https://image.pollinations.ai/prompt/{final_prompt}"
        response = requests.get(url)
        
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))
            image_filename = os.path.join(fname, f"{i + 1}.png")
            image.save(image_filename)
            print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
        else:
            print(f"Error: Failed to generate image {i + 1}")

def generate_images_deepai(prompts, fname):
    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)

    for i, prompt in enumerate(prompts):
        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        
        r = requests.post(
            "https://api.deepai.org/api/text2img",
            data={
                'text': final_prompt,
            },
            headers={'api-key': DEEPAI_API_KEY}
        )
        
        if r.status_code == 200:
            result = r.json()
            if 'output_url' in result:
                image_url = result['output_url']
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image_filename = os.path.join(fname, f"{i + 1}.png")
                    image.save(image_filename)
                    print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                else:
                    print(f"Error: Failed to download image {i + 1}")
            else:
                print(f"Error: No output URL for image {i + 1}")
        else:
            print(f"Error: Failed to generate image {i + 1}")

def generate_images(image_prompts):
    active_folder = str(uuid.uuid4())
    try:
        if IMAGE_SOURCE == "Segmind":
            generate_images_segmind(image_prompts, active_folder)
        elif IMAGE_SOURCE == "Hercai":
            generate_images_hercai(image_prompts, active_folder)
        elif IMAGE_SOURCE == "StableDiffusion":
            generate_images_stable_diffusion(image_prompts, active_folder)
        elif IMAGE_SOURCE == "Prodia":
            generate_images_prodia(image_prompts, active_folder)
        elif IMAGE_SOURCE == "Pollinations":
            generate_images_pollinations(image_prompts, active_folder)
        elif IMAGE_SOURCE == "DeepAI":
            generate_images_deepai(image_prompts, active_folder)
        else:
            raise ValueError("Invalid image generator choice. Please choose a valid option.")
    except Exception as e:
        warning(f"Failed to generate images using {IMAGE_SOURCE}. Error: {str(e)}")
        warning("Falling back to Stable Diffusion for image generation.")
        generate_images_stable_diffusion(image_prompts, active_folder)
    return active_folder

if __name__ == "__main__":
    info("Main block started")
    try:
        niche = "Artificial Intelligence"
        topic = generate_topic(niche)
        script = generate_script(topic, LANGUAGE)
        metadata = generate_metadata(topic, script)
        image_prompts = generate_prompts(script, topic)
        
        success("Summary of generated content:")
        info(f"Topic: {topic}")
        info(f"Script: {script[:100]}...")
        info(f"Metadata - Title: {metadata['title']}")
        info(f"Metadata - Description: {metadata['description'][:100]}...")
        info(f"Number of image prompts: {len(image_prompts)}")
        
        # Generate images
        active_folder = generate_images(image_prompts)
        success(f"Images generated and saved in folder: {active_folder}")
        
        success("Main block finished successfully")
    except Exception as e:
        error(f"An error occurred in the main block: {str(e)}")		
