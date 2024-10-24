import schedule
import subprocess
import sys
import os

from art import *
from cache import *
from utils import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from classes.Tts import TTS
from termcolor import colored
from classes.YouTube import YouTube
from prettytable import PrettyTable
from classes.Outreach import Outreach
from classes.AFM import AffiliateMarketing

def main():
    info("Starting main function")

    valid_input = False
    while not valid_input:
        try:
            info("Displaying options to user")
            info("\n============ OPTIONS ============", False)

            print(colored(" 1. YouTube Automation", "cyan"))
            print(colored(" 2. Affiliate Marketing", "cyan"))
            print(colored(" 3. Outreach", "cyan"))
            print(colored(" 4. Exit", "cyan"))
            print(colored("00. Main Menu", "yellow"))

            info("=================================\n", False)
            user_input = input("Select an option: ").strip()
            if user_input == '':
                warning("Empty input received")
                print("\n" * 100)
                raise ValueError("Empty input is not allowed.")
            if user_input == '00':
                info("Returning to main menu")
                return
            if user_input == '4':
                info("Exiting application")
                sys.exit(0)
            user_input = int(user_input)
            valid_input = True
            info(f"User selected option: {user_input}")
        except ValueError as e:
            error(f"Invalid input: {e}")
            print("\n" * 100)

    if user_input == 1:
        info("Starting YT Shorts Automater...")

        cached_accounts = get_accounts("youtube")
        info(f"Retrieved {len(cached_accounts)} cached YouTube accounts")

        if len(cached_accounts) == 0:
            warning("No accounts found in cache. Prompting to create one.")
            print("\n1. Yes")
            print("2. No")
            print("00. Return to Main Menu")
            user_input = question("Select an option: ")

            if user_input == "1":
                generated_uuid = str(uuid4())
                info(f"Generated new UUID: {generated_uuid}")

                success(f" => Generated ID: {generated_uuid}")
                name = question(" => Enter a name for this account: ")
                profile_path = question(" => Enter the path to the Firefox profile: ")
                niche = question(" => Enter the account niche: ")
                language = question(" => Enter the account language: ")

                add_account("youtube", {
                    "id": generated_uuid,
                    "name": name,
                    "profile_path": profile_path,
                    "niche": niche,
                    "language": language,
                    "videos": []
                })
                success(f"Added new YouTube account: {name}")
            elif user_input == "2" or user_input == "00":
                info("Returning to main menu")
                return
            else:
                error("Invalid option selected. Returning to main menu.")
                return
        else:
            info("Displaying cached YouTube accounts")
            table = PrettyTable()
            table.field_names = ["ID", "UUID", "name", "Niche"]

            for account in cached_accounts:
                table.add_row([cached_accounts.index(account) + 1, colored(account["id"], "cyan"), colored(account["name"], "blue"), colored(account["niche"], "green")])

            print(table)

            user_input = question("Select an account to start (or 00 to return to main menu): ")

            if user_input == "00":
                info("Returning to main menu")
                return

            selected_account = None

            for account in cached_accounts:
                if str(cached_accounts.index(account) + 1) == user_input:
                    selected_account = account

            if selected_account is None:
                error("Invalid account selected. Returning to main menu.")
                return
            else:
                info(f"Selected YouTube account: {selected_account['name']}")
                youtube = YouTube(
                    selected_account["id"],
                    selected_account["name"],
                    selected_account["profile_path"],
                    selected_account["niche"],
                    selected_account["language"]
                )

                while True:
                    rem_temp_files()
                    info("Removed temporary files")
                    info("\n============ OPTIONS ============", False)

                    for idx, youtube_option in enumerate(YOUTUBE_OPTIONS):
                        print(colored(f" {idx + 1}. {youtube_option}", "cyan"))
                    print(colored("00. Return to Main Menu", "cyan"))

                    info("=================================\n", False)

                    user_input = question("Select an option: ")
                    if user_input == "00":
                        info("Returning to main menu")
                        break

                    user_input = int(user_input)
                    info(f"User selected YouTube option: {user_input}")
                    tts = TTS()

                    if user_input == 1:
                        info("Generating YouTube video")
                        youtube.generate_video(tts)
                        print("\n1. Yes")
                        print("2. No")
                        upload_to_yt = question("Do you want to upload this video to YouTube? ")
                        if upload_to_yt == "1":
                            info("Uploading video to YouTube")
                            youtube.upload_video()
                    elif user_input == 2:
                        info("Retrieving YouTube videos")
                        videos = youtube.get_videos()

                        if len(videos) > 0:
                            info(f"Displaying {len(videos)} videos")
                            videos_table = PrettyTable()
                            videos_table.field_names = ["ID", "Date", "Title"]

                            for video in videos:
                                videos_table.add_row([
                                    videos.index(video) + 1,
                                    colored(video["date"], "blue"),
                                    colored(video["title"][:60] + "...", "green")
                                ])

                            print(videos_table)
                        else:
                            warning("No videos found.")
                    elif user_input == 3:
                        info("Setting up CRON job for YouTube uploads")
                        info("How often do you want to upload?")

                        info("\n============ OPTIONS ============", False)
                        for idx, cron_option in enumerate(YOUTUBE_CRON_OPTIONS):
                            print(colored(f" {idx + 1}. {cron_option}", "cyan"))
                        print(colored("00. Return to Previous Menu", "cyan"))

                        info("=================================\n", False)

                        user_input = question("Select an Option: ")
                        if user_input == "00":
                            continue

                        user_input = int(user_input)

                        cron_script_path = os.path.join(ROOT_DIR, "src", "cron.py")
                        command = f"python {cron_script_path} youtube {selected_account['id']}"

                        def job():
                            info("Executing CRON job for YouTube upload")
                            subprocess.run(command)

                        if user_input == 1:
                            info("Setting up daily upload")
                            schedule.every(1).day.do(job)
                            success("Set up CRON Job for daily upload.")
                        elif user_input == 2:
                            info("Setting up twice daily upload")
                            schedule.every().day.at("10:00").do(job)
                            schedule.every().day.at("16:00").do(job)
                            success("Set up CRON Job for twice daily upload.")
                        else:
                            info("Invalid option. Returning to previous menu.")
                    elif user_input == 4:
                        if get_verbose():
                            info(" => Climbing Options Ladder...", False)
                        info("Returning to main menu")
                        break

    elif user_input == 2:
        info("Starting Affiliate Marketing...")

        cached_products = get_products()
        info(f"Retrieved {len(cached_products)} cached products")

        if len(cached_products) == 0:
            warning("No products found in cache. Prompting to create one.")
            print("\n1. Yes")
            print("2. No")
            print("00. Return to Main Menu")
            user_input = question("Select an option: ")

            if user_input == "1":
                affiliate_link = question(" => Enter the affiliate link: ")
                platform = question(" => Enter the platform (e.g., YouTube, Twitter): ")
                account_uuid = question(f" => Enter the {platform} Account UUID: ")

                # Find the account
                account = None
                for acc in get_accounts(platform.lower()):
                    if acc["id"] == account_uuid:
                        account = acc

                add_product({
                    "id": str(uuid4()),
                    "affiliate_link": affiliate_link,
                    "platform": platform,
                    "account_uuid": account_uuid
                })
                success(f"Added new product with affiliate link: {affiliate_link}")

                afm = AffiliateMarketing(affiliate_link, account["profile_path"], account["id"], account["name"], account.get("topic", account.get("niche")))

                info("Generating pitch for affiliate marketing")
                afm.generate_pitch()
                info(f"Sharing pitch on {platform}")
                afm.share_pitch(platform.lower())
            elif user_input == "2" or user_input == "00":
                info("Returning to main menu")
                return
            else:
                error("Invalid option selected. Returning to main menu.")
                return
        else:
            info("Displaying cached products")
            table = PrettyTable()
            table.field_names = ["ID", "Affiliate Link", "Platform", "Account UUID"]

            for product in cached_products:
                table.add_row([cached_products.index(product) + 1, colored(product["affiliate_link"], "cyan"), colored(product["platform"], "blue"), colored(product["account_uuid"], "green")])

            print(table)

            user_input = question("Select a product to start (or 00 to return to main menu): ")

            if user_input == "00":
                info("Returning to main menu")
                return

            selected_product = None

            for product in cached_products:
                if str(cached_products.index(product) + 1) == user_input:
                    selected_product = product

            if selected_product is None:
                error("Invalid product selected. Returning to main menu.")
                return
            else:
                info(f"Selected product with affiliate link: {selected_product['affiliate_link']}")
                # Find the account
                account = None
                for acc in get_accounts(selected_product["platform"].lower()):
                    if acc["id"] == selected_product["account_uuid"]:
                        account = acc

                afm = AffiliateMarketing(selected_product["affiliate_link"], account["profile_path"], account["id"], account["name"], account.get("topic", account.get("niche")))

                info("Generating pitch for affiliate marketing")
                afm.generate_pitch()
                info(f"Sharing pitch on {selected_product['platform']}")
                afm.share_pitch(selected_product["platform"].lower())

    elif user_input == 3:
        info("Starting Outreach...")

        outreach = Outreach()
        outreach.start()
    elif user_input == 4:
        if get_verbose():
            info(" => Quitting...", False)
        info("Exiting application")
        sys.exit(0)
    else:
        error("Invalid option selected. Restarting main function.")
        main()

if __name__ == "__main__":
    info("Starting application")
    print_banner()
    info("Printed ASCII banner")

    first_time = get_first_time_running()
    info(f"First time running: {first_time}")

    if first_time:
        info("First time setup initiated")
        print(colored("Hey! It looks like you're running MoneyPrinter V2 for the first time. Let's get you setup first!", "yellow"))

    info("Setting up file tree")
    assert_folder_structure()

    info("Removing temporary files")
    rem_temp_files()

    info("Fetching MP3 files")
    fetch_Music()

    while True:
        main()
