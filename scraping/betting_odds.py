import os
import sys
import requests
import logging
from datetime import date

import json
import pandas as pd

from git import Git


class Betting_Odds:
    """Scrape betting odds"""

    def __init__(self, logger, season_data):
        """
        Args:
            logger (logging.Logger): Logging pacakge
            season_data (int): Season year
        """
        self.season = season_data['season']

        self.next_gw = self.get_fpl_metadata()

        self.root = f'data/betting/{self.season}-{self.season % 2000 + 1}/{self.next_gw}'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

    def get_fpl_metadata(self):
        """ Request the FPL API

        Returns:
            (int): Next GW
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
        """Scrape historical betting odds"""
        self.logger.info("Loading historical odds ...")
        for season in [2016, 2017, 2018, 2019, 2020, 2021]:
            df = pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{season%2000*1000 + season%2000+1}/E0.csv')
            df.to_csv(self.root + f'/{season}-{season%2000+1}.csv')

    def get_live_odds(self, api_key):
        """ Scrape current betting odds

        Args:
            api_key (string): Secret API key

        Returns:
            json: Betting odds
        """
        self.logger.info("Loading current odds ...")

        odds_response = requests.get(
            'https://api.the-odds-api.com/v3/odds',
            params={
                'api_key': api_key,
                'sport': 'soccer_epl',
                'region': 'uk',  # uk | us | eu | au
                'mkt': 'h2h'  # h2h | spreads | totals
            })
        odds_json = json.loads(odds_response.text)

        with open(f'{self.root}/{date.today()}.json', 'w') as outfile:
            json.dump(odds_json, outfile)

        return odds_json

    def process_odds(self, odds_json):
        """ Data cleaning

        Args:
            odds_json (json): json containing all the betting data

        Returns:
            pd.DataFrame: df containing all the betting data
        """
        df = pd.DataFrame(columns=['Game', 'Site', 'Home', 'Away', 'Draw'])

        def format_game(game, home):
            if game[0] == home:
                return home + ' - ' + game[1], 0, 1
            else:
                return home + ' - ' + game[0], 1, 0

        if odds_json['success']:
            for idx in range(10):
                game = odds_json['data'][idx]['teams']
                home = odds_json['data'][idx]['home_team']

                for site in odds_json['data'][idx]['sites']:
                    frmt, hm_index, aw_index = format_game(game, home)

                    df = df.append({
                        'Game': frmt,
                        'Site': site['site_key'],
                        'Home': site['odds']['h2h'][hm_index],
                        'Away': site['odds']['h2h'][aw_index],
                        'Draw': site['odds']['h2h'][2],
                        }, ignore_index=True)

        df.to_csv(f'{self.root}/{date.today()}/odds.csv')
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    with open('api_key.json') as f:
        api_key = json.load(f)['api_key']

    Betting_Odds(logger, season_data).get_live_odds(api_key)

    if len(sys.argv) > 1:
        logger.info("Saving data ...")
        Git()
    else:
        print("Local")
