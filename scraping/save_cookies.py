# Inspired by https://github.com/JamesEisele/fplr-cookies except that I save cookies instead of sessions
import json
import time
import pickle

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

def fplr_cookies():
    # Load login and password
    with open('fplr_credentials.json') as creds_file:
        creds = json.load(creds_file)

    # Specify Selenium options (Windows OS example).
    chrome_service = Service('/home/pf/Downloads/chromedriver_linux64/chromedriver')
    driver = webdriver.Chrome(service=chrome_service)

    # Log into FPLR
    base_URL = 'https://fplreview.com/'
    driver = fplr_login(base_URL + 'fplr_enter/', driver, creds)

    # Save cookies into pkl file
    session_cookies = driver.get_cookies()
    for cookie in session_cookies:
        if cookie['name'][:19] == 'wordpress_logged_in':   # Only cookie we care about.
            pickle.dump(cookie , open("cookies.pkl", "wb"))
            break

    driver.quit()


def fplr_login(url, driver, creds):
    # Load URL
    driver.get(url)
    # Fill credentials
    driver.find_element(By.XPATH, '//input[@data-key="username"]').send_keys(creds['email'])
    driver.find_element(By.XPATH, '//input[@data-key="user_password"]').send_keys(creds['password'])
    driver.find_element(By.CSS_SELECTOR, ".um-icon-android-checkbox-outline-blank").click() # "Keep me singed in"

    time.sleep(3) # Wait for iframe to load.
    driver.switch_to.frame(driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]'))

    # Wait for user input to complete CAPTCHA, continue after success.
    done = False
    timer = 0 
    while not done:
        time.sleep(1)
        timer += 1

        # Monitor CAPTCHA status.
        element = driver.find_element(By.CLASS_NAME, 'rc-anchor-aria-status')
        if str(element.get_attribute('innerText')) == 'You are verified':
            driver.switch_to.default_content()
            driver.find_element(By.ID, 'um-submit-btn').click() # Login button
            done = True

        elif timer == 180:
            print('Failed to complete CAPTCHA after 180 seconds. Stopping...')
            driver.quit()
            sys.exit(0)

    return driver

if __name__ == '__main__':
   fplr_cookies()

