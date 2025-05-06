
import os
from dotenv import load_dotenv
from .schwab_api import Client


class SchwabInitializer:
    def __init__(self):
        load_dotenv()

        self.app_key = os.getenv('app_key')
        self.app_secret = os.getenv('app_secret')
        self.callback_url = os.getenv('callback_url')

        self.client = Client(self.app_key, self.app_secret, self.callback_url, verbose=True)

    def get_client(self):
        return self.client


def schwab():
    return SchwabInitializer().get_client()
