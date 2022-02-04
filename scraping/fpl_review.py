import os
import sys
import requests
import logging

import json
import pandas as pd
from bs4 import BeautifulSoup
import pickle

from git import Git


class FPL_Review_Scraper:
    """ Scrape FPL Review website """

    def __init__(self, logger, season_data, team_id):
        """
        Args:
            logger (logging.logger): logging package
            season_data (int): Season
            team_id (int): Player team ID
        """
        self.season = season_data['season']

        self.root = f'data/fpl_review/{self.season}-{self.season % 2000 + 1}/gameweek/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.next_gw, self.players = self.get_fpl_metadata()

        self.logger = logger

        self.team_id = team_id

    def get_fpl_metadata(self):
        """ Request the FPL API

        Returns:
            (tuple): Next GW and player ids
        """
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()

        # Get current gameweek
        next_gw = self.get_next_gw(res['events'])
        if not os.path.exists(os.path.join(self.root, str(next_gw))):
            os.mkdir(os.path.join(self.root, str(next_gw)))

        # Get player ids
        cols = ["id", "first_name", "second_name", "team"]
        players = pd.DataFrame(res['elements'])[cols]
        players = players.set_index("id")

        return next_gw, players

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

    def get_free_planner_data(self):
        """Get the FPL Review data"""
        period = min(5, 39 - self.next_gw)
        url = 'https://fplreview.com/free-planner/#forecast_table'
        body = {
            'HiveMind': 'Yes',
            'Weeks': period,
            'TeamID': self.team_id,
        }

        x = requests.post(url, data=body)
        soup = BeautifulSoup(x.content, 'html.parser')

        logger.info("Saving raw data.")
        for fplr_api in soup.find(id="fplr_api"):
            with open(
                    os.path.join(
                        os.path.join(self.root, str(self.next_gw)),
                        'raw_fplreview_fp.json'),
                    'w') as outfile:
                json.dump(json.loads(fplr_api), outfile)

        # Columns
        csv_cols = ["id", "Pos", "Name", "BV", "SV", "Team"]
        for gw in range(self.next_gw, self.next_gw + period):
            csv_cols.append(str(gw) + '_xMins')
            csv_cols.append(str(gw) + '_Pts')

        logger.info("Saving processed data.")
        pd.DataFrame(columns=csv_cols).to_csv(
            os.path.join(
                os.path.join(self.root, str(self.next_gw)),
                'fplreview_fp.csv'),
            index=False)

        for fplr_api in soup.find(id="fplr_api"):
            for idx, key in enumerate(json.loads(fplr_api).keys()):
                try:
                    row = [
                        key,
                        json.loads(fplr_api)[key]['pos'],
                        json.loads(fplr_api)[key]['name'],
                        json.loads(fplr_api)[key]['def_cost'],
                        json.loads(fplr_api)[key]['now_cost'],
                        json.loads(fplr_api)[key]['team_abbrev']
                        ]
                    for gw in range(self.next_gw, self.next_gw + period):
                        row.append(
                            json.loads(fplr_api)[key][str(gw)]['dmins'])
                        row.append(
                            json.loads(fplr_api)[key][str(gw)]['livpts'])

                    (
                        pd.DataFrame([row], columns=csv_cols)
                        .to_csv(
                            os.path.join(
                                os.path.join(self.root, str(self.next_gw)),
                                'fplreview_fp.csv'),
                            index=False, mode='a', header=False))

                except:
                    self.logger.warning(f"Failed to save row {key}.")
                    continue

    def get_free_planner_data_fast(self):
        """Get the FPL Review data"""
        period = min(5, 39 - self.next_gw)
        url = 'https://fplreview.com/free-planner/#forecast_table'
        body = {
            'HiveMind': 'Yes',
            'Weeks': period,
            'TeamID': self.team_id,
        }

        x = requests.post(url, data=body)
        soup = BeautifulSoup(x.content, 'html.parser')

        logger.info("Saving raw data.")
        for fplr_api in soup.find(id="fplr_api"):
            with open(
                    os.path.join(
                        os.path.join(self.root, str(self.next_gw)),
                        'raw_fplreview_fp.json'),
                    'w') as outfile:
                json.dump(json.loads(fplr_api), outfile)

        logger.info("Saving processed data.")

        # Columns
        csv_cols = ["id", "Pos", "Name", "BV", "SV", "Team"]
        df = pd.DataFrame(columns=csv_cols)

        df_json = pd.read_json(
            os.path.join(
                os.path.join(self.root, str(self.next_gw)),
                'raw_fplreview_fp.json')
                ).T

        df[['Pos', 'Name', 'BV', 'SV', 'Team']] = df_json[['pos', 'name', 'def_cost', 'now_cost', 'team_abbrev']]
        df['id'] = df_json.index

        df_json = df_json.reset_index()

        for gw in range(self.next_gw, self.next_gw + period):
            df_gw = pd.json_normalize(df_json[str(gw)]).join(df_json['index'])

            df_gw = df_gw.rename(
                columns={
                    'dmins': f'{gw}_xMins',
                    'livpts': f'{gw}_Pts',
                    'index': 'id'
                    })

            df = pd.merge(
                df,
                df_gw[[f'{gw}_xMins', f'{gw}_Pts', 'id']],
                left_on='id',
                right_on='id',
                )

        df.to_csv(
            os.path.join(
                os.path.join(self.root, str(self.next_gw)),
                'fplreview_fp.csv'),
            index=False)

    def get_premium_planner_data_fast(self):
        """Get the FPL Review data"""
        period = min(8, 39 - self.next_gw)
        url = 'https://fplreview.com/massive-data-planner/#forecast_table'
        body = {
            'HiveMind': 'Yes',
            'Weeks': period,
            'TeamID': self.team_id,
        }

        logger.info("Logging in with cookies.")
        # Get the saved cookies.
        cookies = pickle.load(open("cookies.pkl", "rb"))
        # Set cookies
        session = requests.Session()
        session.cookies.set(cookies['name'], cookies['value'])
        # Request url
        x = session.post(url, data=body)
        soup = BeautifulSoup(x.content, 'html.parser')

        for fplr_api in soup.find(id="fplr_api"):
            with open(
                os.path.join(
                    os.path.join(self.root, str(self.next_gw)),
                    'raw_fplreview_mp.json'),
                    'w') as outfile:
                json.dump(json.loads(fplr_api), outfile)

        logger.info("Processing data.")
        # Columns
        csv_cols = ["id", "Pos", "Name", "BV", "SV", "Team"]
        df = pd.DataFrame(columns=csv_cols)

        df_json = pd.read_json(
            os.path.join(
                os.path.join(self.root, str(self.next_gw)),
                'raw_fplreview_mp.json')
                ).T

        df[['Pos', 'Name', 'BV', 'SV', 'Team']] = df_json[['pos', 'name', 'def_cost', 'now_cost', 'team_abbrev']]
        df['id'] = df_json.index

        df_json = df_json.reset_index()

        for gw in range(self.next_gw, self.next_gw + period):
            df_gw = pd.json_normalize(df_json[str(gw)]).join(df_json['index'])

            df_gw = df_gw.rename(
                columns={
                    'dmins': f'{gw}_xMins',
                    'livpts': f'{gw}_Pts',
                    'index': 'id'
                    })

            df = pd.merge(
                df,
                df_gw[[f'{gw}_xMins', f'{gw}_Pts', 'id']],
                left_on='id',
                right_on='id',
                )

        df.to_csv(
            os.path.join(
                os.path.join(self.root, str(self.next_gw)),
                'fplreview_mp.csv'),
            index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    fplrs = FPL_Review_Scraper(logger, season_data, team_id=35868)
    logger.info("Scraping Planner Data.")

    if sys.argv[1] == 'premium':
        fplrs.get_premium_planner_data_fast()
    elif sys.argv[1] == 'slow':
        fplrs.get_free_planner_data()
    elif sys.argv[1] == 'fast':
        fplrs.get_free_planner_data_fast()

    if len(sys.argv) > 2:
        logger.info("Saving data ...")
        Git()
    else:
        print("Local")
