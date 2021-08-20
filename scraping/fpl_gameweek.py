import os
import requests
import logging
import time 

import json
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from git import Git 

class FPL_Gameweek:
    # Get the Gameweek state
    def __init__(self, logger, season_data):
        self.season = season_data['season']

        self.root = f'data/fpl_official/{self.season}-{self.season % 2000 + 1}/gameweek/'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.current_gw, self.players = self.get_fpl_metadata()
        self.players.to_csv(self.root + f'player_ids.csv')

        self.logger = logger


    def get_fpl_metadata(self):
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()
        
        # Get current gameweek
        current_gw = self.get_current_gw(res['events'])
        if not os.path.exists(self.root + '{current_gw}/'):
            os.mkdir(self.root + '{current_gw}/')
        
        # Get player ids
        cols = ["id", "first_name", "second_name", "team"]
        players = pd.DataFrame(res['elements'])[cols]
        players = players.set_index("id")
        
        return current_gw, players


    def get_current_gw(self, events):
        for idx, gw in enumerate(events):
            if gw['is_current']:
                return idx + 1


    def sample_ranks(self):
        # Data management
        transfer_strategy = self.players.copy()
        transfer_strategy.loc[:, ['Top_100_in', 'Top_1K_in', 'Top_10K_in', 'Top_50K_in', 'Top_100K_in', 'Top_250K_in', 'Top_500K_in',\
        'Top_100_out', 'Top_1K_out', 'Top_10K_out', 'Top_50K_out', 'Top_100K_out', 'Top_250K_out', 'Top_500K_out']] = 0
        self.players.loc[:, ['Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K', 'Top_500K']] = 0
        captain = self.players.copy()
        chip_strategy = pd.DataFrame(index=['wildcard', 'freehit', 'bboost', '3xc'])
        chip_strategy.loc[:, ['Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K', 'Top_500K']] = 0
        hit_strategy = pd.DataFrame(index=['transfers'])
        hit_strategy.loc[:, ['Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K', 'Top_500K']] = 0

        # Sample ~10% of players teams
        range_limits = [
            ('Top_100', 0, 100, 75),
            ('Top_1K', 100, 1000, 500),
            ('Top_10K', 1000, 10000, 2000),
            ('Top_50K', 10000, 50000, 4000),
            ('Top_100K', 50000, 100000, 5000),
            ('Top_250K', 100000, 250000, 15000),
            ('Top_500K', 250000, 500000, 25000)
            ]

        for col, min_rank, max_rank, n_samples in range_limits:
            self.logger.info(f"Starting to scrape {col} ranks")
            fpl_ranks = np.random.randint(min_rank, max_rank, n_samples)

            # Concurrent API Requests
            with ProcessPoolExecutor(max_workers=8) as executor:
                team_data = list(executor.map(self.get_fpl_strategy, fpl_ranks))

            for team, cap, chip, transfer in team_data:
                # Ownership
                for p in team:
                    self.players.loc[p, col] += 1
                captain.loc[cap[0], col] += 1
                # Chip strategy
                if chip is not None:
                    chip_strategy.loc[chip, col] += 1
                # Transfer strategy
                transfer_in, transfer_out = transfer
                for p_in, p_out in zip(transfer_in, transfer_out):
                    transfer_strategy.loc[p_in, col+'_in'] += 1
                    transfer_strategy.loc[p_out, col+'_out'] += 1
                hit_strategy.loc['transfers', col] += len(transfer_in)

            self.players.loc[:, col] = self.players.loc[:, col] / n_samples * 100
            captain.loc[:, col] = captain.loc[:, col] / n_samples * 100
            chip_strategy.loc[:, col] = chip_strategy.loc[:, col] / n_samples * 100
            hit_strategy.loc[:, col] = hit_strategy.loc[:, col] / n_samples
            transfer_strategy.loc[:, col+'_in'] = transfer_strategy.loc[:, col+'_in'] / n_samples * 100
            transfer_strategy.loc[:, col+'_out'] = transfer_strategy.loc[:, col+'_out'] / n_samples * 100

        self.players.to_csv(self.root + f"{self.current_gw}/player_ownership.csv")
        captain.to_csv(self.root + f"{self.current_gw}/captain.csv")
        chip_strategy.to_csv(self.root + f"{self.current_gw}/chip_strategy.csv")
        hit_strategy.to_csv(self.root + f"{self.current_gw}/hit_strategy.csv")
        transfer_strategy.to_csv(self.root + f"{self.current_gw}/transfer_strategy.csv")


    def get_fpl_teamid(self, rank):
        # Scrape the correct page
        page = rank // 50 + 1
        place = rank % 50
        url = f'https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings={page}'
        res = requests.get(url)
        return res.json()['standings']['results'][place]['entry']


    def get_fpl_team(self, team_id):
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{self.current_gw}/picks/').json()
        return [i['element'] for i in res['picks']], [i['element'] for i in res['picks'] if i['multiplier'] > 1]


    def get_fpl_strategy(self, rank):
        attempts = 3
        while attempts:
            try:
                team_id = self.get_fpl_teamid(rank)
                fpl_team, fpl_cap = self.get_fpl_team(team_id)       
                fpl_chips = self.get_fpl_chips(team_id)       
                fpl_transfers = self.get_fpl_transfers(team_id)       
                return fpl_team, fpl_cap, fpl_chips, fpl_transfers
            except:
                attempts -= 1
                if not attempts:
                    self.logger.warning(f"API Call to rank {rank} failed after 3 attempts.")
                    return [], [], [], []

                self.logger.warning(f'API Call failed, retrying in 3 seconds! Rank: {rank}')
                time.sleep(3)


    def get_fpl_chips(self, team_id):
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/history/').json()['chips']
        if res == []:
            return None

        if res[-1]['event'] == self.current_gw:
            return res[-1]['name']
        else:
            return None


    def get_fpl_transfers(self, team_id):
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/transfers').json()
        
        transfer_in = []
        transfer_out = []

        for transfers in res:
            if transfers['event'] == self.current_gw:
                transfer_in.append(transfers['element_in'])
                transfer_out.append(transfers['element_out'])
            else:
                return transfer_in, transfer_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as f:
        season_data = json.load(f)

    fplg = FPL_Gameweek(logger, season_data)
    fplg.sample_ranks()

    logger.info("Saving DataFrame.")
    # Git()