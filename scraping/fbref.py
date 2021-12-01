import os
import requests
import logging

import json
from bs4 import BeautifulSoup
import pandas as pd


class FBRef:
    """Scrape FBRef website"""

    def __init__(self, logger, season_data):
        """
        Args:
            logger (logging.logger): Logging package
            season_data (int): Season
        """
        self.root = 'data/fbref'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

        self.season = season_data['season']

    def get_competition_urls(self, url):
        """ Get all the links of previous EPL seasons

        Returns:
            (list): past url seasons
        """
        res = requests.get(url)
        parsed_html = BeautifulSoup(res.text, 'html.parser')
        past_seasons = []

        for table in parsed_html.findAll('table'):
            for a in table.findAll('a'):
                if 'comps' in a['href'] and a['href'] not in past_seasons:
                    past_seasons.append(a['href'])

        return past_seasons

    def get_fixtures(self):
        """ Scrape data fixture data """

        for index, comp in zip(
                ["9", "690", "514", "8", "19"],
                ['Premier-League', 'EFL-Cup', "FA-Cup", "Champions-League", "Europa-League"]):
            
            # Get links of historical competitions
            seasons = self.get_competition_urls(
                f'https://fbref.com/en/comps/{index}/history/{comp}-Seasons')

            self.logger.info(f"Downloading {comp} Fixtures Data")
            for season in seasons:
                if season.split('/')[-2] == index:
                    url = (
                        f'https://fbref.com/en/comps/{index}/schedule/{comp}-Scores-and-Fixtures')
                    year = self.season

                else:
                    url = (
                        f'https://fbref.com/en/comps/{index}/' +
                        season.split('/')[-2] +
                        '/schedule/' +
                        season.split('/')[-1][:-6] +
                        '-Scores-and-Fixtures')
                    year = season.split('/')[-1][:4]

                # Skip years with no underlying stats
                if int(year) > 2016:
                    self.logger.info(f"Season: {season}")

                    df = pd.read_html(url)[0]
                    # Remove empty row
                    df = df[~(df.Date.isna())]
                    # Add Competition label
                    df["Competition"] = comp

                    if "Wk" in df.columns :
                        if comp == "Champions-League" or comp == "Europa-League":
                            df = df.drop(["Wk"], axis=1)
                        else:
                            df = df.rename(columns={'Wk': "Round"})

                    df = df.loc[:, [
                        "Round", "Day", "Date", "Time", "Home", "Score", "Away",
                        "Attendance", "Venue", "Referee",
                        "Notes", "Competition"]]

                    if os.path.isfile(os.path.join(self.root, 'fixtures.csv')):
                        df.to_csv(
                            os.path.join(self.root, 'fixtures.csv'),
                            index=False, mode='a', header=False)
                    else:
                        df.to_csv(
                            os.path.join(self.root, 'fixtures.csv'),
                            index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as stat:
        season_data = json.load(stat)

    fbref = FBRef(logger, season_data)
    fbref.get_fixtures()