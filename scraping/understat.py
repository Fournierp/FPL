import os
import requests
import logging

import json
from bs4 import BeautifulSoup
import pandas as pd
import re
import codecs
import unidecode


class Understat:
    """Scrape Understat website"""

    def __init__(self, logger, season_data):
        """
        Args:
            logger (logging.logger): Logging package
            season_data (int): Season
        """
        self.root = 'data/understat'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

        self.season = season_data['season']
        self.last_match_id = season_data['match_id']

        self.players = self.get_fpl_metadata()
        self.players.to_csv(self.root + '/fpl_player_ids.csv')

    def get_fpl_metadata(self):
        """ Request the FPL API

        Returns:
            (tuple): Next GW and player ids
        """
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()

        # Get player ids
        cols = ["id", "first_name", "second_name", "team"]
        players = pd.DataFrame(res['elements'])[cols]
        players = players.set_index("id")
        players["name"] = players[
            ['first_name', 'second_name']].apply(lambda x: ' '.join(x), axis=1)
        return players

    def get_url(self, url):
        """ Request the url

        Args:
            url (string): link

        Returns:
            [type]: Parsed html
        """
        res = requests.get(url)
        parsed_html = BeautifulSoup(res.text, 'html.parser')
        return parsed_html.findAll('script')

    def save_season_data(self, output_file, team, player):
        """ Save to CSV

        Args:
            output_file (string): path
            team (json): team data
            player (array): player data
        """
        if not os.path.exists(output_file):
            os.makedirs(output_file)

        new_team_data = []
        for t, v in team.items():
            new_team_data += [v]
        for data in new_team_data:
            team_df = pd.DataFrame.from_records(data["history"])
            team = data["title"].replace(' ', '_')
            team_df.to_csv(
                os.path.join(output_file, team + '.csv'), index=False)
        player_df = pd.DataFrame.from_records(player)
        player_df.to_csv(
            os.path.join(output_file, 'players.csv'), index=False)

    def get_raw_season_data(self, history=False):
        """ Scrape season data

        Args:
            history (bool, optional): Get past data. Defaults to False.
        """
        if history:
            seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
        else:
            seasons = [self.season]
        self.logger.info("Downloading Historical Season Data")

        for season in seasons:
            self.logger.info(f"Season: {season}")
            scripts = self.get_url(
                f"https://understat.com/league/EPL/{season}")
            team = {}
            player = {}
            for script in scripts:
                for c in script.contents:
                    split_data = c.split('=')
                    var_name = split_data[0].strip()
                    if var_name == 'var teamsData':
                        content = re.findall(
                            r'JSON\.parse\(\'(.*)\'\)',
                            split_data[1])
                        decoded_content = codecs.escape_decode(
                            content[0], "hex")[0].decode('utf-8')
                        team = json.loads(decoded_content)
                    elif var_name == 'var playersData':
                        content = re.findall(
                            r'JSON\.parse\(\'(.*)\'\)',
                            split_data[1])
                        decoded_content = codecs.escape_decode(
                            content[0], "hex")[0].decode('utf-8')
                        player = json.loads(decoded_content)

            self.save_season_data(
                self.root + f'/{season}-{season % 2000 + 1}', team, player)

    def save_match_data(self, output_file, shots, player):
        """ Save to CSV

        Args:
            output_file (string): path
            shots (json): shots data
            player (array): player data
        """
        if not os.path.exists(output_file):
            os.makedirs(output_file)

        shot_df = pd.DataFrame(shots['h'])
        shot_df = shot_df.append(pd.DataFrame(shots['a']))
        shot_df.to_csv(output_file + '/shots.csv', index=False)

        player_df = pd.DataFrame.from_dict(player['h'], orient='index')
        player_df = player_df.append(
            pd.DataFrame.from_dict(player['a'], orient='index'))
        player_df.to_csv(output_file + '/players.csv', index=False)

    def get_match_data(self):
        """Scrape match data"""
        url = 'https://understat.com/match/'
        league_name = 'EPL'
        last_valid_match_id = 0

        for i in range(self.last_match_id, self.last_match_id + 500):
            res = requests.get(url + f'{i}')
            parsed_html = BeautifulSoup(res.text, 'html.parser')
            # Get Game description
            desc = parsed_html.findAll("ul", class_="breadcrumb")
            # Confirm the game is in EPL league
            for league in desc:
                if league.find(text=re.compile(league_name)):
                    self.logger.info(f"Downloading Match: {i} Data")
                    # Extract the match's data
                    scripts = parsed_html.findAll('script')
                    shots = {}
                    player = {}
                    for script in scripts:
                        for c in script.contents:
                            split_data = c.split('=')
                            var_name = split_data[0].strip()
                            if var_name == 'var shotsData':
                                content = re.findall(
                                    r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                                decoded_content = codecs.escape_decode(
                                    content[0], "hex")[0].decode('utf-8')
                                shots = json.loads(decoded_content)
                            elif var_name == 'var rostersData':
                                content = re.findall(
                                    r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                                decoded_content = codecs.escape_decode(
                                    content[0], "hex")[0].decode('utf-8')
                                player = json.loads(decoded_content)

                    team_home = parsed_html.findAll(
                        "div", class_="roster roster-home")
                    for team in team_home:
                        team_h = team.text.strip().partition('\n')[0]

                    team_away = parsed_html.findAll(
                        "div", class_="roster roster-away")
                    for team in team_away:
                        team_a = team.text.strip().partition('\n')[0]

                    self.save_match_data(
                        self.root + f'/matches/{league.text.strip()[-11:]}\
                            _{team_h}_vs_{team_a}', shots, player)

                if (league.find(text=re.compile(league_name)) or
                        league.find(text=re.compile('La liga')) or
                        league.find(text=re.compile('Bundesliga')) or
                        league.find(text=re.compile('Serie A')) or
                        league.find(text=re.compile('RFPL')) or
                        league.find(text=re.compile('Ligue 1'))):
                    # Record url was scraped
                    last_valid_match_id = i

        with open('info.json') as f:
            season_data = json.load(f)

        with open('info.json', 'w') as f:
            season_data['match_id'] = last_valid_match_id
            json.dump(season_data, f)

    def _lambda_req(self, row):
        """ Request url

        Args:
            row (array): [description]

        Returns:
            [type]: [description]
        """
        name = row['name']
        name = unidecode.unidecode(name)
        print(name)

        url = f'https://understat.com/main/getPlayersName/{name}'
        res = requests.get(url.format(name=name))

        if res.status_code == 200:
            players = json.loads(res.text)['response']['players']

            if len(players) == 1:
                return players[0]['id']
        else:
            name = row['second_name']
            name = unidecode.unidecode(name)
            res = requests.get(url.format(name=name))

            if res.status_code == 200:
                players = json.loads(res.text)['response']['players']

                if len(players) == 1:
                    return [player['id'] for player in players]

            return -1

    def get_fpl_to_understat_ids(self):
        '''Buggy for complex, long names.'''
        self.players['id'] = self.players.apply(
            lambda row: self._lambda_req(row), axis=1)
        self.players.to_csv(self.root + '/understat_ids.csv')

    def save_player_data(self, output_file, groups, shots, matches):
        """ Save player data to disk

        Args:
            output_file (string): path
            groups ([type]): [description]
            shots (json): shots data
            matches (array): matches data
        """
        if not os.path.exists(output_file):
            os.makedirs(output_file)

        seasons_df = pd.DataFrame.from_dict(groups['season'])
        seasons_df.to_csv(output_file + '/seasons.csv', index=False)

        shot_df = pd.DataFrame(shots)
        shot_df.to_csv(output_file + '/shots.csv', index=False)

        matches_df = pd.DataFrame.from_records(matches)
        matches_df.to_csv(output_file + '/matches.csv', index=False)

    def get_player_data(self, history=False):
        if history:
            seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
        else:
            seasons = [self.season]

        url = 'https://understat.com/player/'

        for season in seasons:
            if not os.path.exists(f'data/understat/{season}/'):
                os.makedirs(f'data/understat/{season}/')

            df = pd.read_csv(f'data/understat/{season}/players.csv')
            player_ids = df['id']

            for idx in player_ids:
                self.logger.info(f"Downloading Player: {idx} Data")
                res = requests.get(url + f'{idx}')
                parsed_html = BeautifulSoup(res.text, 'html.parser')
                # Extract the match's data
                scripts = parsed_html.findAll('script')
                groups = {}
                shots = {}
                matches = {}
                # Get Player name
                desc = parsed_html.find("ul", class_="breadcrumb")
                name = list(desc.stripped_strings)[-1]

                for script in scripts:
                    for c in script.contents:
                        split_data = c.split('=')
                        var_name = split_data[0].strip()
                        if var_name == 'var groupsData':
                            content = re.findall(
                                r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                            decoded_content = codecs.escape_decode(
                                content[0], "hex")[0].decode('utf-8')
                            groups = json.loads(decoded_content)
                        elif var_name == 'var shotsData':
                            content = re.findall(
                                r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                            decoded_content = codecs.escape_decode(
                                content[0], "hex")[0].decode('utf-8')
                            shots = json.loads(decoded_content)
                        elif var_name == 'var matchesData':
                            content = re.findall(
                                r'JSON\.parse\(\'(.*)\'\)', split_data[1])
                            decoded_content = codecs.escape_decode(
                                content[0], "hex")[0].decode('utf-8')
                            matches = json.loads(decoded_content)

                self.save_player_data(
                    self.root + f'/players/{name}', groups, shots, matches)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    understat = Understat(logger, season_data)
    # understat.get_raw_season_data()
    understat.get_match_data()
    # understat.get_player_data(True)
    # understat.get_player_data()
    # understat.get_fpl_to_understat_ids()
