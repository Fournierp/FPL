import os
import sys
import requests
import logging

import json
import pandas as pd

from git import Git


class FiveThirtyEight:
    """Scrape FiveThirtyEight Soccer Perforance Index"""

    def __init__(self, logger, season_data):
        """
        Args:
            logger (logging.logger): Logging package
            season_data (int): Season of interest
        """
        self.season = season_data['season']

        self.root = f'data/fivethirtyeight/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.next_gw = self.get_fpl_metadata()

        self.logger = logger

    def get_fpl_metadata(self):
        """ Request the FPL API

        Returns:
            (tuple): Next GW and player ids
        """
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()

        # Get current gameweek
        return self.get_next_gw(res['events'])

    def get_next_gw(self, events):
        """ Get the next gameweek to be played in the EPL

        Args:
            events (json): FPL API response

        Returns:
            (int): Next gameweek
        """
        for idx, gw in enumerate(events):
            if gw['is_next']:
                return idx + 1

    def get_historical_data(self):
        """Scrape historical SPI"""
        self.logger.info("Loading spi_matches ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
        df.to_csv(self.root + f'/spi_matches.csv', index=False)

        self.logger.info("Loading spi_matches_latest ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv')
        df.to_csv(self.root + f'/spi_matches_latest.csv', index=False)

        self.logger.info("Loading spi_global_rankings ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv')
        df.to_csv(self.root + f'/spi_global_rankings.csv', index=False)

    def update_ranking_data(self):
        """Scrape current SPI"""
        self.logger.info("Loading spi_global_rankings ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv')

        if not os.path.exists(self.root + f'{self.season}-{self.season % 2000 + 1}/{self.next_gw}'):
            os.makedirs(self.root + f'{self.season}-{self.season % 2000 + 1}/{self.next_gw}')

        df.to_csv(self.root + f'{self.season}-{self.season % 2000 + 1}/{self.next_gw}/spi_global_rankings.csv', index=False)

        self.logger.info("Loading spi_matches_latest ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv')
        df.to_csv(self.root + f'spi_matches_latest.csv', index=False)

        self.logger.info("Loading spi_matches ...")
        df = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv')
        df.to_csv(self.root + f'spi_matches.csv', index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    fivethirtyeight = FiveThirtyEight(logger, season_data)
    # fivethirtyeight.get_historical_data()
    fivethirtyeight.update_ranking_data()

    if len(sys.argv) > 1:
        logger.info("Saving data ...")
        Git()
    else:
        print("Local")
