import os
import requests
import logging

import json
from bs4 import BeautifulSoup
import pandas as pd
import re
import codecs


class Understat:

    def __init__(self, logger, season_data):
        self.root = 'data/understat'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

        self.season = season_data['season']

        self.players = self.get_fpl_metadata()
        self.players.to_csv(self.root + '/fpl_player_ids.csv')


    def get_fpl_metadata(self):
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()
        
        # Get player ids
        cols = ["id", "first_name", "second_name", "team"]
        players = pd.DataFrame(res['elements'])[cols]
        players = players.set_index("id")
        players["name"] = players[['first_name', 'second_name']].apply(lambda x: ' '.join(x), axis=1)
        return players


    def get_url(self, url):
        res = requests.get(url)
        parsed_html = BeautifulSoup(res.text, 'html.parser')
        return parsed_html.findAll('script')


    def save_season_data(self, output_file, team, player):
        if not os.path.exists(output_file):
            os.makedirs(output_file)

        new_team_data = []
        for t, v in team.items():
            new_team_data += [v]
        for data in new_team_data:
            team_df = pd.DataFrame.from_records(data["history"])
            team = data["title"].replace(' ', '_')
            team_df.to_csv(os.path.join(output_file, team + '.csv'), index=False)
        player_df = pd.DataFrame.from_records(player)
        player_df.to_csv(os.path.join(output_file, 'players.csv'), index=False)


    def get_raw_season_data(self, history=False):
        if history:
            seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
        else :
            seasons = [self.season]
        self.logger.info("Downloading Historical Season Data")
        url = 'https://understat.com/league/EPL'

        for season in seasons:
            self.logger.info(f"Season: {season}")
            scripts = self.get_url(f"https://understat.com/league/EPL/{season}")
            team = {}
            player = {}
            for script in scripts:
                for c in script.contents:
                    split_data = c.split('=')
                    var_name = split_data[0].strip()
                    if var_name == 'var teamsData':
                        content = re.findall(r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                        decoded_content = codecs.escape_decode(content[0], "hex")[0].decode('utf-8')
                        team = json.loads(decoded_content)
                    elif var_name == 'var playersData':
                        content = re.findall(r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                        decoded_content = codecs.escape_decode(content[0], "hex")[0].decode('utf-8')
                        player = json.loads(decoded_content)

            self.save_season_data(self.root + f'/{season}-{season % 2000 + 1}', team, player)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    understat = Understat(logger, season_data)
    understat.get_raw_season_data()
