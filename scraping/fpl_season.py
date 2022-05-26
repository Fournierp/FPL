import os
import sys, getopt
import requests
import logging
import time
from tqdm import tqdm

import json
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from git import Git


class FPL_Season:
    """Get the top 250K FPL managers' season strategy"""

    def __init__(self, logger, season_data, argv):
        """
        Args:
            logger (logging.logger): logging package
            season_data (int): Season
            argv ([type]): CLI arguments
        """
        self.season = season_data['season']

        self.root = f'data/fpl_official/{self.season}-{self.season % 2000 + 1}/season/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.players = self.get_fpl_metadata()

        self.logger = logger

        self.get_cli_args(argv)

    def get_cli_args(self, argv):
        """ Get CLI arguments

        Args:
            argv ([type]): CLI arguments
        """
        self.git_cli = False
        try:
            opts, args = getopt.getopt(argv, "g:", ["git="])
        except getopt.GetoptError:
            print('Usage is : script.py -g <boolean>')

        for opt, arg in opts:
            if opt in ("-g", "--git"):
                self.git_cli = arg

        if self.git_cli:
            self.logger.info("The script will push files.")
        else:
            self.logger.info("The script will not push files.")

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

        return players

    def sample_ranks(self):
        """Sample every rank to get season data"""
        last = 95000
        end = 150000
        increment = 5000
        for ranks in np.arange(last + increment, end, increment):
            self.logger.info(f"Starting to scrape top {ranks}.")
            managers = {}
            fpl_ranks = np.arange(last, ranks)
            flag_rep = np.repeat(True, increment)

            # Concurrent API Requests
            with ProcessPoolExecutor(max_workers=8) as executor:
                team_data = list(
                    executor.map(self.get_fpl_strategy, fpl_ranks, flag_rep))

            for (rank, team_id, chips, overall_rank, bench_pts,
                    team, transfers) in team_data:
                managers[str(rank)] = {}
                managers[str(rank)]['id'] = team_id
                managers[str(rank)]['chips'] = chips
                managers[str(rank)]['team'] = team['team']
                managers[str(rank)]['cap'] = team['cap']
                managers[str(rank)]['vice'] = team['vice']
                managers[str(rank)]['subs'] = team['subs']
                managers[str(rank)]['transfers'] = transfers
                managers[str(rank)]['overall_rank'] = overall_rank
                managers[str(rank)]['bench_pts'] = bench_pts

            with open(
                    os.path.join(self.root, f'managers_{ranks}.json'),
                    'w') as outfile:
                json.dump(managers, outfile)

            if self.git_cli:
                self.logger.info(f"Saving DataFrame after {ranks} API Calls.")
                Git()

            last = ranks

    def sample_one_manager(self, team_id):
        """ Get one manager's season data

        Args:
            team_id (int): FPL manager team id
        """
        self.logger.info(f"Starting to scrape manager {team_id}.")
        managers = {}

        # API Requests
        team_data = self.get_fpl_strategy(team_id, False)

        (rank, team_id, chips, overall_rank,
            bench_pts, team, transfers) = team_data
        managers[str(rank)] = {}
        managers[str(rank)]['id'] = team_id
        managers[str(rank)]['chips'] = chips
        managers[str(rank)]['team'] = team['team']
        managers[str(rank)]['cap'] = team['cap']
        managers[str(rank)]['vice'] = team['vice']
        managers[str(rank)]['subs'] = team['subs']
        managers[str(rank)]['transfers'] = transfers
        managers[str(rank)]['overall_rank'] = overall_rank
        managers[str(rank)]['bench_pts'] = bench_pts

        with open(
                os.path.join(self.root, f'manager_id_{team_id}.json'),
                'w') as outfile:
            json.dump(managers, outfile)

        if self.git_cli:
            self.logger.info("Saving csv.")
            Git()

    def get_fpl_teamid(self, rank):
        """ Get the FPL Team ID based on the rank

        Args:
            rank (int): Manager rank

        Returns:
            int: FPL Team ID
        """
        page = rank // 50 + 1
        place = rank % 50
        url = f'https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings={page}'
        res = requests.get(url)
        return res.json()['standings']['results'][place]['entry']

    def get_fpl_hist(self, team_id):
        """ Get season data

        Args:
            team_id (int): FPL manager id

        Returns:
            (tuple): chips, overall rank, bench points
        """
        res = requests.get(
            f'https://fantasy.premierleague.com/api/entry/{team_id}/history/'
            ).json()
        chips_res = res['chips']
        current_res = res['current']

        chips = {
            'wildcard_1': 0
        }

        for chip in chips_res:
            if chip['name'] == 'wildcard' and not chips['wildcard_1']:
                chips['wildcard_1'] = chip['event']
            elif chip['name'] == 'freehit':
                chips['freehit'] = chip['event']
            elif chip['name'] == 'bboost':
                chips['bboost'] = chip['event']
            elif chip['name'] == '3xc':
                chips['threexc'] = chip['event']
            elif chip['name'] == 'wildcard':
                chips['wildcard_2'] = chip['event']

        overall_rank = {}

        for gameweek in current_res:
            overall_rank[gameweek['event']] = gameweek['overall_rank']

        bench_pts = {}

        for gameweek in current_res:
            bench_pts[gameweek['event']] = gameweek['points_on_bench']

        return chips, overall_rank, bench_pts

    def get_fpl_team(self, team_id):
        """ Get FPL team player ids

        Args:
            team_id (int): FPL Manager id

        Returns:
            dict: Data
        """
        data = {
            'team': {},
            'cap': {},
            'vice': {},
            'subs': {}
        }

        for gw in range(1, 39):
            res = requests.get(
                f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/'
                ).json()
            if len(res) == 1:
                self.logger.warning(f'Team: {team_id} did not play in \
                    GW: {gw} ?!')
                data['team'][str(gw)] = []
                data['cap'][str(gw)] = []
                data['vice'][str(gw)] = []
                data['subs'][str(gw)] = []
            else:
                data['team'][str(gw)] = [i['element'] for i in res['picks']]
                data['cap'][str(gw)] = [
                    i['element'] for i in res['picks'] if i['is_captain']][0]
                data['vice'][str(gw)] = [
                    i['element'] for i in res['picks']
                    if i['is_vice_captain']][0]
                if len(res['automatic_subs']):
                    data['subs'][str(gw)] = [
                        (i['element_in'], i['element_out'])
                        for i in res['automatic_subs']]
        return data

    def get_fpl_transfers(self, team_id):
        """ Get FPL team transfers

        Args:
            team_id (int): FPL Manager id

        Returns:
            dict: Data
        """
        data = {}
        res = requests.get(
            f'https://fantasy.premierleague.com/api/entry/{team_id}/transfers'
            ).json()

        for gameweek in res:
            if str(gameweek['event']) not in data.keys():
                data[str(gameweek['event'])] = {}
                data[str(gameweek['event'])]['in'] = {}
                data[str(gameweek['event'])]['out'] = {}

            data[str(gameweek['event'])]['in'][len(
                data[str(gameweek['event'])]['in'])] = gameweek['element_in']
            data[str(gameweek['event'])]['out'][len(
                data[str(gameweek['event'])]['out'])] = gameweek['element_out']

        return data

    def get_fpl_strategy(self, rank, flag):
        """ Get FPL team data

        Args:
            rank (int): FPL manager rank

        Returns:
            (tuple): data
        """
        if not (rank + 1) % 1000:
            self.logger.warning(f"Done with {rank} ranks.")

        attempts = 3
        while attempts:
            try:
                if flag:
                    team_id = self.get_fpl_teamid(rank)
                else:
                    team_id = rank
                chips, overall_rank, bench_pts = self.get_fpl_hist(team_id)
                team = self.get_fpl_team(team_id)
                transfers = self.get_fpl_transfers(team_id)
                return (rank, team_id, chips, overall_rank,
                        bench_pts, team, transfers)
            except:
                attempts -= 1
                if not attempts:
                    self.logger.warning(
                        f"API Call to rank {rank} failed after 3 attempts.")
                    data = {
                        'team': {},
                        'cap': {},
                        'vice': {},
                        'subs': {}
                    }
                    return [], [], [], [], [], data, []

                self.logger.warning(
                    f'API Call failed, retrying in 3 seconds! Rank: {rank}')
                time.sleep(3)

    def sample_hof(self):
        """ Get one manager's season data """
        df = pd.read_csv('data/hof/top_managers.csv')

        managers = {}
        for team_id in tqdm(df['team_id']):

            # API Requests
            res = requests.get(
                f'https://fantasy.premierleague.com/api/entry/{team_id}/history/'
                ).json()
            rank = res['current'][23]['overall_rank']

            team_data = self.get_fpl_strategy(team_id, False)

            (_, team_id, chips, overall_rank,
                bench_pts, team, transfers) = team_data
            managers[str(rank)] = {
                'id': team_id,
                'chips': chips,
                'team': team['team'],
                'cap': team['cap'],
                'vice': team['vice'],
                'subs': team['subs'],
                'transfers': transfers,
                'overall_rank': overall_rank,
                'bench_pts': bench_pts
                }

            with open(
                    os.path.join(self.root, f'managers_hof.json'),
                    'w') as outfile:
                json.dump(managers, outfile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    fpls = FPL_Season(logger, season_data, sys.argv[1:])
    fpls.sample_ranks()
    # fpls.sample_one_manager(team_id=35868)
    # fpls.sample_hof()
