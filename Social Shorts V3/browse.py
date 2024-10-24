"""Gets the browser's given the user's input"""
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.edge.options import Options as EdgeOptions

# Webdriver managers
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service as EdgeService

from selenium import webdriver

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from termcolor import colored
import logging
import os
from datetime import datetime
import sys
import platform

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/browser_automation_{timestamp}.log'

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

class Browser:
    def __init__(self):
        self.driver = None
        self.system = platform.system()
        logger.info(f"Browser automation initialized on {self.system} system")

    def get_default_profile_path(self, browser_name):
        """Get default browser profile paths based on operating system"""
        system = self.system
        user_home = os.path.expanduser('~')
        
        profile_paths = {
            'windows': {
                'chrome': f'{user_home}\\AppData\\Local\\Google\\Chrome\\User Data\\Default',
                'firefox': f'{user_home}\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles',
                'edge': f'{user_home}\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default',
            },
            'darwin': {  # macOS
                'chrome': f'{user_home}/Library/Application Support/Google/Chrome/Default',
                'firefox': f'{user_home}/Library/Application Support/Firefox/Profiles',
                'safari': f'{user_home}/Library/Safari',
                'edge': f'{user_home}/Library/Application Support/Microsoft Edge/Default',
            },
            'linux': {
                'chrome': f'{user_home}/.config/google-chrome/Default',
                'firefox': f'{user_home}/.mozilla/firefox',
                'edge': f'{user_home}/.config/microsoft-edge/Default',
            }
        }

        system_lower = system.lower()
        if system_lower in profile_paths:
            return profile_paths[system_lower].get(browser_name.lower())
        return None

    def get_browser(self, name='chrome', headless=False, profile_path=None):
        logger.info(f"Initializing {name} browser (headless: {headless})")
        
        try:
            if name.lower() == 'chrome':
                options = ChromeOptions()
                if headless:
                    options.add_argument('--headless')
                if profile_path:
                    options.add_argument(f'user-data-dir={profile_path}')
                self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
                logger.info(f"Chrome browser initialized with profile: {profile_path}")

            elif name.lower() == 'firefox':
                options = FirefoxOptions()
                if headless:
                    options.add_argument('--headless')
                if profile_path:
                    options.add_argument('-profile')
                    options.add_argument(profile_path)
                self.driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)
                logger.info(f"Firefox browser initialized with profile: {profile_path}")

            elif name.lower() == 'safari':
                options = SafariOptions()
                # Safari doesn't support profile paths in the same way
                self.driver = webdriver.Safari(options=options)
                logger.info("Safari browser initialized")

            elif name.lower() == 'edge':
                options = EdgeOptions()
                if headless:
                    options.add_argument('--headless')
                if profile_path:
                    options.add_argument(f'user-data-dir={profile_path}')
                self.driver = webdriver.Edge(EdgeChromiumDriverManager().install(), options=options)
                logger.info(f"Edge browser initialized with profile: {profile_path}")

            else:
                logger.error(f"Unsupported browser: {name}")
                raise ValueError("Unsupported browser")

            return self.driver

        except Exception as e:
            logger.error(f"Error initializing {name} browser: {str(e)}")
            raise

    def close_browser(self):
        if self.driver:
            logger.info("Closing browser")
            self.driver.quit()

def main():
    browser = Browser()
    options = [
        "Open Chrome",
        "Open Firefox",
        "Open Safari",
        "Open Edge",
        "Exit"
    ]

    while True:
        print(colored("\nBrowser Options:", "cyan"))
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        choice = input(colored("\nEnter your choice (1-5): ", "yellow"))
        logger.info(f"User selected option: {choice}")

        if choice == "5":
            logger.info("User chose to exit")
            print(colored("Exiting...", "green"))
            break

        if choice in ["1", "2", "3", "4"]:
            headless = input(colored("Run in headless mode? (y/n): ", "yellow")).lower() == 'y'
            use_profile = input(colored("Use browser profile? (y/n): ", "yellow")).lower() == 'y'
            
            browser_names = {
                "1": "chrome",
                "2": "firefox",
                "3": "safari",
                "4": "edge"
            }
            
            selected_browser = browser_names[choice]
            profile_path = None

            if use_profile:
                default_profile = browser.get_default_profile_path(selected_browser)
                print(colored(f"Default profile path: {default_profile}", "cyan"))
                custom_profile = input(colored("Enter custom profile path (or press Enter for default): ", "yellow"))
                profile_path = custom_profile if custom_profile else default_profile
                logger.info(f"Using profile path: {profile_path}")

            try:
                driver = browser.get_browser(selected_browser, headless, profile_path)
                url = input(colored("Enter the URL to navigate to: ", "yellow"))
                logger.info(f"Navigating to URL: {url}")
                driver.get(url)
                print(colored(f"Navigated to {url}", "green"))
                
                input(colored("Press Enter to close the browser...", "yellow"))
                browser.close_browser()
                
            except Exception as e:
                logger.error(f"Error during browser operation: {str(e)}")
                print(colored(f"An error occurred: {str(e)}", "red"))

        else:
            logger.warning("Invalid choice entered")
            print(colored("Invalid choice. Please try again.", "red"))

if __name__ == "__main__":
    main()
