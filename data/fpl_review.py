import os
import requests
import logging
import time 

import json
import csv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from concurrent.futures import ProcessPoolExecutor

from git import Git


class FPL_Review_Scraper:

    def __init__(self, logger, season_data, team_id):
        self.season = season_data['season']

        self.root = f'data/fpl_review/{self.season}-{self.season % 2000 + 1}/gameweek/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        self.next_gw, self.players = self.get_fpl_metadata()

        self.logger = logger

        self.team_id = team_id


    def get_fpl_metadata(self):
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()
        
        # Get current gameweek
        next_gw = self.get_next_gw(res['events'])
        if not os.path.exists(self.root + str(next_gw) + '/'):
            os.mkdir(self.root + str(next_gw) + '/')
        
        # Get player ids
        cols = ["id", "first_name", "second_name", "team"]
        players = pd.DataFrame(res['elements'])[cols]
        players = players.set_index("id")
    
        return next_gw, players


    def get_next_gw(self, events):
        for idx, gw in enumerate(events):
            if gw['is_next']:
                return idx + 1


    def get_free_planner_data(self, raw=True):
        period = min(5, 39 - self.next_gw)
        url = 'https://fplreview.com/free-planner/#forecast_table'
        body = {
            'HiveMind': 'Yes',
            'Weeks': period,
            'TeamID': self.team_id,
        }

        x = requests.post(url, data=body)
        soup = BeautifulSoup(x.content, 'html.parser')

        with open(self.root + str(self.next_gw) + '/fplreview_fp.csv', 'w', newline='', encoding="utf-8") as fplr_file:
            writer = csv.writer(fplr_file)
            # Columns
            csv_cols = ["id", "Pos", "Name", "BV", "SV", "Team"]
            for gw in range(self.next_gw, self.next_gw + period):
                csv_cols.append(str(gw)+ '_xMins')
                csv_cols.append(str(gw)+ '_Pts')
            writer.writerow(csv_cols)
            # Players
            for fplr_api in soup.find(id="fplr_api"):
                logger.info("Saving raw data.")
                with open(self.root + str(self.next_gw) + '/raw_fplreview_fp.json', 'w') as outfile:
                    json.dump(json.loads(fplr_api), outfile)

                logger.info("Saving processed data.")
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
                            row.append(json.loads(fplr_api)[key][str(gw)]['dmins'])
                            row.append(json.loads(fplr_api)[key][str(gw)]['livpts'])
                        writer.writerow(row)
                    except:
                        self.logger.warning(f"Failed to save row {key}.")
                        continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    fplrs = FPL_Review_Scraper(logger, season_data, team_id=35868)
    logger.info("Scraping Free Planner Data.")
    fplrs.get_free_planner_data()

    logger.info("Saving data ...")
    Git()
    