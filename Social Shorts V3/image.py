import os
import time
import random
import requests
import io
import logging
from PIL import Image
from termcolor import colored
import openai
from datetime import datetime
import sys

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/image_generator_{timestamp}.log'

    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

logger = setup_logging()

class ImageGenerator:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        self.segmind_api_key = os.getenv('SEGMIND_API_KEY')
        self.prodia_api_key = os.getenv('PRODIA_API_KEY')
        self.deepai_api_key = os.getenv('DEEPAI_API_KEY')
        self.image_model = "v3"  # Default model for Hercai
        logger.info("ImageGenerator initialized with available APIs")
        
        # Log available APIs
        available_apis = []
        if self.openai_api_key: available_apis.append("DALLE")
        if self.stability_api_key: available_apis.append("Stable Diffusion")
        if self.segmind_api_key: available_apis.append("Segmind")
        if self.prodia_api_key: available_apis.append("Prodia")
        if self.deepai_api_key: available_apis.append("DeepAI")
        logger.info(f"Available APIs: {', '.join(available_apis)}")

    def dalle(self, prompt):
        logger.info(f"Generating image with DALLE. Prompt: {prompt[:50]}...")
        try:
            openai.api_key = self.openai_api_key
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response['data'][0]['url']
            saved_file = self.save_image_from_url(image_url, "dalle_output.png")
            logger.info(f"DALLE image generated successfully: {saved_file}")
            return saved_file
        except Exception as e:
            logger.error(f"Error generating DALLE image: {str(e)}")
            raise

    def stable_diffusion(self, prompt):
        logger.info(f"Generating image with Stable Diffusion. Prompt: {prompt[:50]}...")
        try:
            stability_api = client.StabilityInference(
                key=self.stability_api_key,
                verbose=True,
            )
            answers = stability_api.generate(
                prompt=prompt,
                seed=992446758,
                steps=30,
                cfg_scale=8.0,
                width=512,
                height=512,
                samples=1,
                sampler=generation.SAMPLER_K_DPMPP_2M
            )
            
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        img.save("stable_diffusion_output.png")
                        logger.info("Stable Diffusion image generated successfully")
                        return "stable_diffusion_output.png"
                        
        except Exception as e:
            logger.error(f"Error generating Stable Diffusion image: {str(e)}")
            raise

    def segmind(self, prompts, fname):
        logger.info(f"Starting Segmind batch generation for {len(prompts)} prompts")
        url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
        headers = {'x-api-key': self.segmind_api_key}

        if not os.path.exists(fname):
            os.makedirs(fname)
            logger.info(f"Created directory: {fname}")

        num_images = len(prompts)
        requests_made = 0
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{num_images}")
            
            if requests_made >= 5 and time.time() - start_time <= 60:
                time_to_wait = 60 - (time.time() - start_time)
                logger.info(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds")
                time.sleep(time_to_wait)
                requests_made = 0
                start_time = time.time()

            try:
                final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
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
                        logger.info(f"Successfully saved image {i + 1}/{num_images} as '{image_filename}'")
                    else:
                        logger.error(f"Invalid image data received for prompt {i + 1}")
                else:
                    logger.error(f"Failed to generate image {i + 1}. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")

            except Exception as e:
                logger.error(f"Error processing prompt {i + 1}: {str(e)}")

    def hercai(self, prompts, fname):
        logger.info(f"Starting Hercai batch generation for {len(prompts)} prompts")
        if not os.path.exists(fname):
            os.makedirs(fname)
            logger.info(f"Created directory: {fname}")

        num_images = len(prompts)
        currentseed = random.randint(1, 1000000)
        logger.info(f"Using seed: {currentseed}")

        for i, prompt in enumerate(prompts):
            try:
                final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
                url = f"https://hercai.onrender.com/{self.image_model}/text2image?prompt={final_prompt}"
                logger.info(f"Processing prompt {i+1}/{num_images}")
                
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
                            logger.info(f"Successfully saved image {i + 1}/{num_images} as '{image_filename}'")
                        else:
                            logger.error(f"Failed to retrieve image from URL for image {i + 1}")
                    else:
                        logger.error(f"No image URL in response for image {i + 1}")
                else:
                    logger.error(f"Failed API request for image {i + 1}. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")

            except Exception as e:
                logger.error(f"Error processing prompt {i + 1}: {str(e)}")

    def prodia(self, prompts, fname):
        logger.info(f"Starting Prodia batch generation for {len(prompts)} prompts")
        if not os.path.exists(fname):
            os.makedirs(fname)
            logger.info(f"Created directory: {fname}")

        num_images = len(prompts)
        headers = {
            "X-Prodia-Key": self.prodia_api_key
        }

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{num_images}")
            try:
                final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
                
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
                    logger.info(f"Job created with ID: {job_id}")
                    
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
                                logger.info(f"Successfully saved image {i + 1}/{num_images} as '{image_filename}'")
                                break
                        elif status["status"] == "failed":
                            logger.error(f"Job failed for image {i + 1}")
                            break
                        
                        time.sleep(1)
                else:
                    logger.error(f"Failed to start job for image {i + 1}. Status code: {response.status_code}")

            except Exception as e:
                logger.error(f"Error processing prompt {i + 1}: {str(e)}")

    def pollinations(self, prompts, fname):
        logger.info(f"Starting Pollinations batch generation for {len(prompts)} prompts")
        if not os.path.exists(fname):
            os.makedirs(fname)
            logger.info(f"Created directory: {fname}")

        num_images = len(prompts)

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{num_images}")
            try:
                final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
                
                url = f"https://image.pollinations.ai/prompt/{final_prompt}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    image_data = response.content
                    image = Image.open(io.BytesIO(image_data))
                    image_filename = os.path.join(fname, f"{i + 1}.png")
                    image.save(image_filename)
                    logger.info(f"Successfully saved image {i + 1}/{num_images} as '{image_filename}'")
                else:
                    logger.error(f"Failed to generate image {i + 1}. Status code: {response.status_code}")

            except Exception as e:
                logger.error(f"Error processing prompt {i + 1}: {str(e)}")

    def deepai(self, prompts, fname):
        logger.info(f"Starting DeepAI batch generation for {len(prompts)} prompts")
        if not os.path.exists(fname):
            os.makedirs(fname)
            logger.info(f"Created directory: {fname}")

        num_images = len(prompts)

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{num_images}")
            try:
                final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
                
                r = requests.post(
                    "https://api.deepai.org/api/text2img",
                    data={
                        'text': final_prompt,
                    },
                    headers={'api-key': self.deepai_api_key}
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
                            logger.info(f"Successfully saved image {i + 1}/{num_images} as '{image_filename}'")
                        else:
                            logger.error(f"Failed to download image {i + 1}")
                    else:
                        logger.error(f"No output URL for image {i + 1}")
                else:
                    logger.error(f"Failed to generate image {i + 1}. Status code: {r.status_code}")

            except Exception as e:
                logger.error(f"Error processing prompt {i + 1}: {str(e)}")

    def save_image_from_url(self, url, filename):
        logger.info(f"Saving image from URL to {filename}")
        try:
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            img.save(filename)
            logger.info(f"Successfully saved image to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving image from URL: {str(e)}")
            raise

def main():
    generator = ImageGenerator()
    logger.info("Image Generator initialized. Starting main application...")
    
    # Add your main application logic here
    
if __name__ == "__main__":
    main()
