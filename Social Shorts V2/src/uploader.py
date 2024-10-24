import os
import json
import time
import random
from typing import List
from datetime import datetime
from selenium import webdriver

from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

from constants import *
from utils import *

from selenium.common.exceptions import TimeoutException, ElementNotInteractableException, NoSuchElementException
import os
import json
import time
import random
from typing import List
from datetime import datetime
from selenium import webdriver

from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

from constants import *
from utils import *

from selenium.common.exceptions import TimeoutException, ElementNotInteractableException, NoSuchElementException

class Uploader:
    def __init__(self, profile_path: str):
        info("Setting up profile for Uploader")
        browser = get_browser() 
        
        if browser == 'firefox':
            self.options = FirefoxOptions()
            if get_headless():
                info("Setting Firefox to headless mode")
                self.options.add_argument("--headless")
            if profile_path:
                info(f"Setting Firefox profile path: {profile_path}")
                self.options.add_argument("-profile")
                self.options.add_argument(profile_path)
            info("Installing GeckoDriver")
            self.service = FirefoxService(GeckoDriverManager().install())
            info("Initializing Firefox browser")
            try:
                self.browser = webdriver.Firefox(service=self.service, options=self.options)
            except Exception as e:
                error(f"Failed to initialize Firefox with custom profile: {e}")
                info("Attempting to initialize Firefox without custom profile")
                self.options = FirefoxOptions()
                if get_headless():
                    self.options.add_argument("--headless")
                self.browser = webdriver.Firefox(service=self.service, options=self.options)
        
        elif browser == 'chrome':
            self.options = ChromeOptions()
            if get_headless():
                info("Setting Chrome to headless mode")
                self.options.add_argument("--headless")
                self.options.add_argument("--disable-gpu")
                self.options.add_argument("--window-size=1920,1080")
            if profile_path:
                info(f"Setting Chrome profile path: {profile_path}")
                self.options.add_argument(f"user-data-dir={profile_path}")
            self.options.add_argument("--no-sandbox")
            self.options.add_argument("--disable-dev-shm-usage")
            info("Installing ChromeDriver")
            self.service = ChromeService(ChromeDriverManager().install())
            info("Initializing Chrome browser")
            try:
                self.browser = webdriver.Chrome(service=self.service, options=self.options)
            except Exception as e:
                error(f"Failed to initialize Chrome with custom profile: {e}")
                info("Attempting to initialize Chrome without custom profile")
                self.options = ChromeOptions()
                if get_headless():
                    self.options.add_argument("--headless")
                    self.options.add_argument("--disable-gpu")
                    self.options.add_argument("--window-size=1920,1080")
                self.browser = webdriver.Chrome(service=self.service, options=self.options)

        elif browser == 'edge':
            self.options = EdgeOptions()
            if get_headless():
                info("Setting Edge to headless mode")
                self.options.add_argument("--headless")
            if profile_path:
                info(f"Setting Edge profile path: {profile_path}")
                self.options.add_argument(f"user-data-dir={profile_path}")
            info("Installing EdgeDriver")
            self.service = EdgeService(EdgeChromiumDriverManager().install())
            info("Initializing Edge browser")
            try:
                self.browser = webdriver.Edge(service=self.service, options=self.options)
            except Exception as e:
                error(f"Failed to initialize Edge with custom profile: {e}")
                info("Attempting to initialize Edge without custom profile")
                self.options = EdgeOptions()
                if get_headless():
                    self.options.add_argument("--headless")
                self.browser = webdriver.Edge(service=self.service, options=self.options)

        elif browser == 'safari':
            self.options = SafariOptions()
            info("Safari does not support user profiles or headless mode in the same way as other browsers.")
            if profile_path:
                info(f"Profile path provided for Safari, but it may not be used: {profile_path}")
            if get_headless():
                info("Headless mode requested for Safari, but it may not be supported.")
            info("Initializing Safari browser")
            try:
                self.browser = webdriver.Safari(options=self.options)
            except Exception as e:
                error(f"Failed to initialize Safari: {e}")
                raise

        else:
            error_message = f"Unsupported browser type: {browser}"
            error(error_message)
            raise ValueError(error_message)

        success(f"{browser.capitalize()} browser setup complete.")

        
        self.channel_id = None
        self.uploaded_video_url = None
        self._account_uuid = None
        self.screenshot_counter = 0
        self.channel_id = None
        self.uploaded_video_url = None
        self._account_uuid = None
        self.screenshot_counter = 0
        self.screenshot_dir = os.path.join(os.getcwd(), 'screenshots')
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def take_screenshot(self, action_name):
        self.screenshot_counter += 1
        screenshot_filename = f"screenshot_{self.screenshot_counter}_{action_name}.png"
        screenshot_path = os.path.join(self.screenshot_dir, screenshot_filename)
        
        try:
            self.browser.save_screenshot(screenshot_path)
            info(f"Screenshot taken: {screenshot_path}")
        except Exception as e:
            info(f"Failed to take screenshot: {str(e)}")

    

    def click_element(self, xpath, timeout=20, skip_if_not_found=False):
        """Wait for an element to be clickable and then click it."""
        try:
            info(f"Attempting to click element with xpath: {xpath}")
            element = WebDriverWait(self.browser, timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            time.sleep(random.uniform(0.5, 1))
            element.click()
            info(f"Successfully clicked element: {xpath}")
            return True
        except TimeoutException:
            if skip_if_not_found:
                warn(f"Element not found, skipping: {xpath}")
                return False
            else:
                error(f"Timeout waiting for element: {xpath}")
                return False
        except Exception as e:
            error(f"Error clicking element {xpath}: {e}")
            return False

    def get_channel_id(self) -> str:
        try:
            info("Getting Channel ID from YouTube Studio")
            driver = self.browser
            driver.get("https://studio.youtube.com")
            info("Navigated to YouTube Studio")
            time.sleep(2)
            channel_id = driver.current_url.split("/")[-1]
            self.channel_id = channel_id
            success(f"Retrieved Channel ID: {channel_id}")
            return channel_id
        except Exception as e:
            error(f"Failed to get Channel ID: {e}")
            return None

    def upload_video(self, video_path, title, description):
        try:
            info("Starting multi-platform video upload process")
            self.get_channel_id()

            # YouTube Upload
            self.upload_to_youtube(video_path, title, description)

            # LinkedIn Upload
            self.upload_to_linkedin(video_path, title, description)

            # Instagram Upload
            self.upload_to_instagram(video_path, title, description)

            # TikTok Upload
            self.upload_to_tiktok(video_path, title, description)

            # Snapchat Upload
            self.upload_to_snapchat(video_path, title, description)

            # Close the browser
            info("Closing browser")
            self.browser.quit()

            return True
        except Exception as e:
            error(f"Failed to upload video: {e}")
            self.browser.quit()
            return False

    def upload_to_youtube(self, video_path, title, description):
        try:
            info("Starting YouTube upload process")
            driver = self.browser
            verbose = get_verbose()

            driver.get("https://www.youtube.com/upload")
            info("Navigated to YouTube upload page")
            self.take_screenshot("youtube_upload_page")

            info("Locating file picker element")
            file_picker = driver.find_element(By.TAG_NAME, "ytcp-uploads-file-picker")
            info("Locating file input element")
            file_input = file_picker.find_element(By.TAG_NAME, "input")
            info(f"Selecting video file: {video_path}")
            file_input.send_keys(video_path)
            info(f"Selected video file: {video_path}")
            self.take_screenshot("youtube_file_selected")

            time.sleep(5)

            try:
                info("Locating title input field")
                title_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="textbox"]'))
                )
                info("Clearing existing title")
                title_box.clear()
                info(f"Entering video title: {title}")
                title_box.send_keys(title)
                info("Title entered successfully")
                self.take_screenshot("youtube_title_entered")
            except Exception as e:
                error(f"Title input failed: {e}")

            try:
                info("Locating description input field")
                description_box = driver.find_elements(By.XPATH, '//*[@id="textbox"]')[1]
                info(f"Entering video description")
                description_box.send_keys(description)
                info("Description entered successfully")
                self.take_screenshot("youtube_description_entered")
            except Exception as e:
                error(f"Description input failed: {e}")

            time.sleep(0.5)

            if verbose:
                info("Setting 'made for kids' option...")
            info("Locating 'made for kids' checkboxes")
            is_for_kids_checkbox = driver.find_element(By.NAME, YOUTUBE_MADE_FOR_KIDS_NAME)
            is_not_for_kids_checkbox = driver.find_element(By.NAME, YOUTUBE_NOT_MADE_FOR_KIDS_NAME)

            if not get_is_for_kids():
                info("Setting video as not made for kids")
                is_not_for_kids_checkbox.click()
            else:
                info("Setting video as made for kids")
                is_for_kids_checkbox.click()
            self.take_screenshot("youtube_kids_option_set")

            time.sleep(0.5)

            for i in range(3):
                if verbose:
                    info(f"Clicking next... (Step {i+1}/3)")
                info(f"Locating next button (Step {i+1}/3)")
                next_button = driver.find_element(By.ID, YOUTUBE_NEXT_BUTTON_ID)
                next_button.click()
                time.sleep(2)
                self.take_screenshot(f"youtube_next_button_{i+1}")

            if verbose:
                info("Setting as unlisted...")
            info("Locating visibility radio buttons")
            radio_button = driver.find_elements(By.XPATH, YOUTUBE_RADIO_BUTTON_XPATH)
            radio_button[2].click()
            success("Set video visibility to unlisted")
            self.take_screenshot("youtube_visibility_set")

            if verbose:
                info("Clicking done button...")
            info("Locating done button")
            done_button = driver.find_element(By.ID, YOUTUBE_DONE_BUTTON_ID)
            done_button.click()
            self.take_screenshot("youtube_upload_done")

            time.sleep(2)

            if verbose:
                info("Getting video URL...")
            info(f"Navigating to YouTube Studio channel page")
            driver.get(f"https://studio.youtube.com/channel/{self.channel_id}/videos/short")
            time.sleep(2)
            info("Locating uploaded videos")
            videos = driver.find_elements(By.TAG_NAME, "ytcp-video-row")
            first_video = videos[0]
            info("Extracting video URL")
            anchor_tag = first_video.find_element(By.TAG_NAME, "a")
            href = anchor_tag.get_attribute("href")
            if verbose:
                info(f"Extracting video ID from URL: {href}")
            video_id = href.split("/")[-2]

            url = build_url(video_id)
            self.uploaded_video_url = url

            if verbose:
                success(f"Uploaded Video: {url}")

            self.add_video({
                "title": title,
                "description": description,
                "url": url,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            return True
        except Exception as e:
            error(f"Failed to upload video to YouTube: {e}")
            return False

    def upload_to_linkedin(self, video_path, title, description):
        info("Starting LinkedIn upload process")
        try:
            driver = self.browser
            driver.get("https://www.linkedin.com/feed/")
            info("Navigated to LinkedIn feed")
            self.take_screenshot("linkedin_feed")

            info("Locating 'Start a post' button")
            start_post_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Start a post']"))
            )
            start_post_button.click()
            self.take_screenshot("linkedin_start_post")

            info("Locating file input element")
            file_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, '//input[@type="file"]'))
            )
            info(f"Uploading video file: {video_path}")
            file_input.send_keys(video_path)
            info(f"Video file uploaded: {video_path}")
            self.take_screenshot("linkedin_file_uploaded")

            info("Locating post input field")
            post_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, '//div[@role="textbox"]'))
            )
            full_post = f"{title}\n\n{description}"
            info("Entering post text")
            post_input.send_keys(full_post)
            info("Post text entered")
            self.take_screenshot("linkedin_post_text")

            info("Locating post button")
            post_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Post']"))
            )
            post_button.click()
            self.take_screenshot("linkedin_post_button")

            info("Waiting for post confirmation")
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Post successful')]"))
            )
            success("LinkedIn upload completed successfully")
            self.take_screenshot("linkedin_upload_success")
        except Exception as e:
            error(f"Error during LinkedIn upload: {e}")

    def upload_to_instagram(self, video_path, title, description):
        info("Starting Instagram upload process")
        try:
            driver = self.browser
            driver.get("https://www.instagram.com/")
            info("Navigated to Instagram homepage")
            self.take_screenshot("instagram_homepage")

            info("Locating 'New post' button")
            new_post_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//*[@aria-label='New post']"))
            )
            new_post_button.click()
            self.take_screenshot("instagram_new_post")

            info("Locating file input element")
            file_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, '//input[@type="file"]'))
            )
            info(f"Uploading video file: {video_path}")
            file_input.send_keys(video_path)
            info(f"Video file uploaded: {video_path}")
            self.take_screenshot("instagram_file_uploaded")

            info("Waiting for video processing")
            time.sleep(15)  # Wait for video processing

            info("Clicking through 'Next' buttons")
            next_buttons = driver.find_elements(By.XPATH, "//*[text()='Next']")
            for i, next_button in enumerate(next_buttons):
                next_button.click()
                time.sleep(2)
                self.take_screenshot(f"instagram_next_{i+1}")

            info("Locating description input field")
            description_input = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//*[@aria-label='Write a caption...']"))
            )
            full_description = f"{title}\n\n{description}"
            info("Entering description")
            description_input.send_keys(full_description)
            info("Description entered")
            self.take_screenshot("instagram_description")

            info("Locating share button")
            share_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//*[text()='Share']"))
            )
            share_button.click()
            self.take_screenshot("instagram_share")

            info("Waiting for upload confirmation")
            WebDriverWait(driver, 120).until(
                lambda d: "Your reel has been shared." in d.page_source
            )
            success("Instagram upload completed successfully")
            self.take_screenshot("instagram_upload_success")
        except Exception as e:
            error(f"Error during Instagram upload: {e}")

    def upload_to_tiktok(self, video_path, title, description):
        info("Starting TikTok upload process")
        try:
            driver = self.browser
            driver.get("https://www.tiktok.com/upload")
            info("Navigated to TikTok upload page")
            self.take_screenshot("tiktok_upload_page")

            info("Locating file input element")
            file_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
            )
            info(f"Uploading video file: {video_path}")
            file_input.send_keys(video_path)
            info(f"Video file uploaded: {video_path}")
            self.take_screenshot("tiktok_file_uploaded")

            info("Locating caption input field")
            caption_input = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'DraftEditor-editorContainer')]//div[contains(@class, 'public-DraftEditor-content')]"))
            )
            full_description = f"{title}\n\n{description}"
            info("Entering description")
            try:
                caption_input.click()
                ActionChains(driver).move_to_element(caption_input).send_keys(full_description).perform()
            except ElementNotInteractableException:
                info("Using JavaScript to set description")
                driver.execute_script("arguments[0].textContent = arguments[1];", caption_input, full_description)
            info("Description entered")
            self.take_screenshot("tiktok_description")

            info("Locating post button")
            post_button = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'TUXButton') and .//div[text()='Post']]"))
            )
            post_button.click()
            self.take_screenshot("tiktok_post_button")

            info("Waiting for upload confirmation")
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.XPATH, "//div[text()='Your video has been uploaded']"))
            )
            success("TikTok upload completed successfully")
            self.take_screenshot("tiktok_upload_success")
        except Exception as e:
            error(f"Error during TikTok upload: {e}")

    def upload_to_snapchat(self, video_path, title, description):
        info("Starting Snapchat upload process")
        try:
            driver = self.browser
            driver.get('https://my.snapchat.com/')
            info("Navigated to Snapchat")
            self.take_screenshot("snapchat_homepage")

            info("Locating file input element")
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file'][accept='video/mp4,video/quicktime,video/webm,image/jpeg,image/png']"))
            )
            info(f"Uploading video file: {video_path}")
            file_input.send_keys(video_path)
            info("Video file uploaded")
            self.take_screenshot("snapchat_file_uploaded")
            time.sleep(10)

            info("Locating post button")
            post_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div/main/div[2]/div[2]/div[2]/div[5]/div[1]/div[1]/div/div[2]/div/div/div/div[1]/div/div/div/div[1]"))
            )
            post_button.click()
            info("Clicked post button")
            self.take_screenshot("snapchat_post_button")

            info("Locating description textarea")
            description_textarea = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Add a description and #topics']"))
            )
            full_description = f"{title}\n\n{description}"
            info("Entering description")
            description_textarea.send_keys(full_description)
            info("Added description")
            self.take_screenshot("snapchat_description")

            info("Locating accept button")
            accept_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div/div[2]/div/div[2]/div[3]/div/button[2]"))
            )
            accept_button.click()
            info("Accepted terms")
            self.take_screenshot("snapchat_accept_terms")

            info("Locating final post button")
            post_final_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Post to Snapchat')]"))
            )
            post_final_button.click()
            info("Clicked final post button")
            self.take_screenshot("snapchat_final_post")

            info("Waiting for upload confirmation")
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.XPATH, "//div[text()='Yay! Your post is now live!']"))
            )
            success("Snapchat upload completed successfully")
            self.take_screenshot("snapchat_upload_success")
        except Exception as e:
            error(f"Error during Snapchat upload: {e}")

    def add_video(self, video: dict) -> None:
        info(f"Adding video to cache: {video}")
        videos = self.get_videos()
        videos.append(video)

        cache = get_youtube_cache_path()

        info("Reading existing cache")
        with open(cache, "r") as file:
            previous_json = json.loads(file.read())
            
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    account["videos"].append(video)
            
            info("Writing updated cache")
            with open(cache, "w") as f:
                f.write(json.dumps(previous_json))
        success("Video added to cache")

    def get_videos(self) -> List[dict]:
        try:
            info("Retrieving uploaded videos from cache")
            if not os.path.exists(get_youtube_cache_path()):
                info("Cache file does not exist, creating new one")
                # Create the cache file
                with open(get_youtube_cache_path(), 'w') as file:
                    json.dump({
                        "accounts": [{
                            "id": self._account_uuid,
                            "videos": []
                        }]
                    }, file, indent=4)
                return []

            videos = []
            info("Reading cache file")
            # Read the cache file
            with open(get_youtube_cache_path(), 'r') as file:
                previous_json = json.loads(file.read())
                # Find our account
                accounts = previous_json.get("accounts", [])
                for account in accounts:
                    if account["id"] == self._account_uuid:
                        videos = account.get("videos", [])
                        break
                
                if not videos:
                    info("Account not found in cache, creating new entry")
                    # If account not found, create a new one
                    new_account = {
                        "id": self._account_uuid,
                        "videos": []
                    }
                    accounts.append(new_account)
                    with open(get_youtube_cache_path(), 'w') as f:
                        json.dump({"accounts": accounts}, f, indent=4)

            success(f"Retrieved {len(videos)} videos from cache")
            return videos
        except Exception as e:
            error(f"Failed to retrieve videos from cache: {e}")
            return []
