import os
import requests
import logging

import json
import pandas as pd
from zipfile import ZipFile

from concurrent.futures import ProcessPoolExecutor

from git import Git


class FiveThirtyEight:

    def __init__(self, logger, season_data):
        self.root = 'data/fivethirtyeight'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

        self.season = season_data['season']


    def get_current_gw(self):
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()
    
        # Get current FPL gameweek
        for idx, gw in enumerate(res['events']):
            if gw['is_current']:
                return idx + 1


    def get_historical_data(self):
        self.logger.info("Loading spi_matches ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
        df.to_csv(self.root + f'/spi_matches.csv')

        self.logger.info("Loading spi_matches_latest ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv')
        df.to_csv(self.root + f'/spi_matches_latest.csv')

        self.logger.info("Loading spi_global_rankings ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv')
        df.to_csv(self.root + f'/spi_global_rankings.csv')


    def update_ranking_data(self):
        self.logger.info("Loading spi_global_rankings ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv')

        current_gw = self.get_current_gw()

        if not os.path.exists(self.root + f'/{self.season}-{self.season % 2000 + 1}/{current_gw}'):
            os.makedirs(self.root + f'/{self.season}-{self.season % 2000 + 1}/{current_gw}')

        df.to_csv(self.root + f'/{self.season}-{self.season % 2000 + 1}/{current_gw}/spi_global_rankings.csv')

        self.logger.info("Loading spi_matches_latest ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv')
        df.to_csv(self.root + f'/spi_matches_latest.csv')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    fivethirtyeight = FiveThirtyEight(logger, season_data)
    # fivethirtyeight.get_historical_data()
    fivethirtyeight.update_ranking_data()

    logger.info("Saving data ...")
    Git()