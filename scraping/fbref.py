import os
import requests
import logging

import json
from bs4 import BeautifulSoup
import pandas as pd
import re

# Squad Standard Stats
stats = [
    "players_used", "possession", "games", "goals", "assists", "goals_pens",
    "pens_made", "pens_att", "cards_yellow", "cards_red", "goals_per90",
    "assists_per90", "goals_assists_per90", "goals_pens_per90",
    "goals_assists_pens_per90", "xg", "npxg", "xa", "xg_per90", "xa_per90",
    "xg_xa_per90", "npxg_per90", "npxg_xa_per90"]

# Squad Goalkeeping
keepers = [
    "gk_used", "goals_against_gk", "goals_against_per90_gk",
    "shots_on_target_against", "saves", "save_pct", "clean_sheets",
    "clean_sheets_pct", "pens_att_gk", "pens_allowed", "pens_saved",
    "pens_missed_gk", "pens_save_pct"]

# Player Goalkeeping
player_keepers = [
    "nationality", "squad", "age", "birth_year",
    "games_gk", "games_starts_gk", "minutes_gk", "goals_against_gk",
    "goals_against_per90_gk", "shots_on_target_against", "saves", "save_pct",
    "wins_gk", "draws_gk", "losses_gk", "clean_sheets", "clean_sheets_pct",
    "pens_att_gk", "pens_allowed", "pens_saved", "pens_missed_gk"]

# Squad Advanced Goalkeeping
keepersadv = [
    "goals_against_gk", "pens_allowed", "free_kick_goals_against_gk",
    "corner_kick_goals_against_gk", "own_goals_against_gk", "psxg_gk",
    "psnpxg_per_shot_on_target_against", "psxg_net_gk", "psxg_net_per90_gk",
    "passes_completed_launched_gk", "passes_launched_gk",
    "passes_pct_launched_gk", "passes_gk", "passes_throws_gk",
    "pct_passes_launched_gk", "passes_length_avg_gk", "goal_kicks",
    "pct_goal_kicks_launched", "goal_kick_length_avg", "crosses_gk",
    "crosses_stopped_gk", "crosses_stopped_pct_gk",
    "def_actions_outside_pen_area_gk", "def_actions_outside_pen_area_per90_gk",
    "avg_distance_def_actions_gk"]

# Squad Shooting
shooting = [
    "shots_total", "shots_on_target", "shots_on_target_pct",
    "shots_total_per90", "shots_on_target_per90", "goals_per_shot",
    "goals_per_shot_on_target", "average_shot_distance", "shots_free_kicks",
    "pens_att", "pens_made"]

# Squad Passing
passing = [
    "passes_completed", "passes", "passes_pct", "passes_total_distance",
    "passes_progressive_distance", "passes_completed_short", "passes_short",
    "passes_pct_short", "passes_completed_medium", "passes_medium",
    "passes_pct_medium", "passes_completed_long", "passes_long",
    "passes_pct_long", "assists", "xa_net", "assisted_shots",
    "passes_into_final_third", "passes_into_penalty_area",
    "crosses_into_penalty_area", "progressive_passes"]

# Squad Pass Types
passing_types = [
    "passes", "passes_live", "passes_dead", "passes_free_kicks",
    "through_balls", "passes_pressure", "passes_switches", "crosses",
    "corner_kicks", "corner_kicks_in", "corner_kicks_out",
    "corner_kicks_straight", "passes_ground", "passes_low", "passes_high",
    "passes_left_foot", "passes_right_foot", "passes_head", "throw_ins",
    "passes_other_body", "passes_completed", "passes_offsides", "passes_oob",
    "passes_intercepted", "passes_blocked"]

# Squad Goal and Shot Creation
gca = [
    "sca", "sca_per90", "sca_passes_live", "sca_passes_dead", "sca_dribbles",
    "sca_shots", "sca_fouled", "gca", "gca_per90", "gca_passes_live",
    "gca_passes_dead", "gca_dribbles", "gca_shots", "gca_fouled",
    "gca_defense"]

# Squad Defensive Actions
defense = [
    "tackles", "tackles_won", "tackles_def_3rd", "tackles_mid_3rd",
    "tackles_att_3rd", "dribble_tackles", "dribbles_vs", "dribble_tackles_pct",
    "dribbled_past", "pressures", "pressure_regains", "pressure_regain_pct",
    "pressures_def_3rd", "pressures_mid_3rd", "pressures_att_3rd", "blocks",
    "blocked_shots", "blocked_shots_saves", "blocked_passes", "interceptions",
    "tackles_interceptions", "clearances", "errors"]

# Squad Possession
possession = [
    "touches", "touches_def_pen_area", "touches_def_3rd", "touches_mid_3rd",
    "touches_att_3rd", "touches_att_pen_area", "touches_live_ball",
    "dribbles_completed", "dribbles", "dribbles_completed_pct",
    "players_dribbled_past", "nutmegs", "carries", "carry_distance",
    "carry_progressive_distance", "progressive_carries",
    "carries_into_final_third", "carries_into_penalty_area", "miscontrols",
    "dispossessed", "pass_targets", "passes_received", "passes_received_pct",
    "progressive_passes_received"]

# Squad Miscellaneous Stats
misc = [
    "cards_yellow", "cards_red", "cards_yellow_red", "fouls", "fouled",
    "offsides", "crosses", "interceptions", "tackles_won", "pens_won",
    "pens_conceded", "own_goals", "ball_recoveries", "aerials_won",
    "aerials_lost", "aerials_won_pct"]


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

    def get_url(self, url):
        """ Request the url

        Args:
            url (string): link

        Returns:
            [type]: Parsed html
        """
        res = requests.get(url)
        # Handle hidden table.
        comm = re.compile("<!--|-->")
        soup = BeautifulSoup(comm.sub("", res.text), 'lxml')
        return soup.findAll("tbody")

    def get_team_table(self, url, columns):
        """ Parse the table of team data

        Args:
            url (string): Link
            columns (list): Column names to select

        Returns:
            pd.DataFrame: Data
        """
        tables = self.get_url(url)
        table_rows = tables[0].find_all('tr')
        team_dict = dict()

        for row in table_rows:
            if row.find('th', {"scope": "row"}) is not None:
                team_name = (
                    row.find('th', {"data-stat": "squad"})
                    .text.strip().encode().decode("utf-8"))

                # Add squad name
                if 'squad' in team_dict:
                    team_dict['squad'].append(team_name)
                else:
                    team_dict['squad'] = [team_name]

                # Parse the table
                for col in columns:
                    # Get the statistic
                    cell = row.find("td", {"data-stat": col})
                    if cell is None:
                        # Fill na
                        text = 'None'
                    else:
                        a = cell.text.strip().encode()
                        text = a.decode("utf-8")
                        # Fill na
                        if text == '':
                            text = '0'

                        text = float(text.replace(',', ''))

                    # Add statistics
                    if col in team_dict:
                        team_dict[col].append(text)
                    else:
                        team_dict[col] = [text]

        return pd.DataFrame.from_dict(team_dict)

    def get_team_data(self, url):
        """ Scrape each table of team data

        Args:
            url (string): Base link to the team stats

        Returns:
            (pd.DataFrame): Final data
        """
        categories_name = [
            'stats', 'keepers', 'keepersadv', 'shooting', 'passing',
            'passing_types', 'gca', 'defense', 'possession', 'misc']
        categories_cols = [
            stats, keepers, keepersadv, shooting, passing, passing_types,
            gca, defense, possession, misc]
        df = []
        for name, cols in zip(categories_name, categories_cols):
            df.append(self.get_team_table(url.format(table=name), cols))

        return pd.concat(df, axis=1)

    def get_player_table(self, url, columns):
        """ Parse the table of goalkeeper data

        Args:
            url (string): Link
            columns (list): Column names to select

        Returns:
            pd.DataFrame: Data
        """
        tables = self.get_url(url)
        player_rows = tables[2].find_all('tr')
        player_dict = dict()

        for row in player_rows:
            if row.find('th', {"scope": "row"}) is not None:
                player_name = (
                    row.find('td', {"data-stat": "player"})
                    .text.strip().encode().decode("utf-8"))

                # Add player name
                if 'player' in player_dict:
                    player_dict['player'].append(player_name)
                else:
                    player_dict['player'] = [player_name]

                # Parse the table
                for col in columns:
                    # Get the statistic
                    cell = row.find("td", {"data-stat": col})
                    if cell is None:
                        # Fill na
                        text = 'None'
                    else:
                        a = cell.text.strip().encode()
                        text = a.decode("utf-8")
                        # Fill na
                        if(text == ''):
                            text = '0'

                        if (
                                (col != 'player') & (col != 'nationality') &
                                (col != 'position') & (col != 'squad') &
                                (col != 'age') & (col != 'birth_year')):
                            print(col)
                            text = float(text.replace(',', ''))

                    if col in player_dict:
                        player_dict[col].append(text)
                    else:
                        player_dict[col] = [text]

        return pd.DataFrame.from_dict(player_dict)

    def get_keeper_data(self, url):
        """ Scrape each table of Goalkeeper data

        Args:
            url (string): Base link to the team stats

        Returns:
            (pd.DataFrame): Final data
        """
        categories_name = ['keepers', 'keepersadv']
        categories_cols = [player_keepers, keepersadv]
        df = []
        for name, cols in zip(categories_name, categories_cols):
            df.append(self.get_player_table(url.format(table=name), cols))

        df = pd.concat(df, axis=1)
        return df.loc[:, ~df.columns.duplicated()]

    def get_pl_urls(self):
        """ Get all the links of previous EPL seasons

        Returns:
            (list): past url seasons
        """
        url = 'https://fbref.com/en/comps/9/history/Premier-League-Seasons'
        res = requests.get(url)
        parsed_html = BeautifulSoup(res.text, 'html.parser')
        past_seasons = []

        for table in parsed_html.findAll('table'):
            for a in table.findAll('a'):
                if 'comps' in a['href'] and a['href'] not in past_seasons:
                    past_seasons.append(a['href'])

        return past_seasons

    def get_pl_season(self, history=False):
        """ Scrape data fron seasons

        Args:
            history (bool, optional): Scrape current. Defaults to False.
        """
        seasons = self.get_pl_urls()

        if not history:
            seasons = [seasons[0]]

        self.logger.info("Downloading Historical Season Data")

        for season in seasons:
            self.logger.info(f"Season: {season}")

            if season.split('/')[-2] == '9':
                url = (
                    'https://fbref.com/en/comps/9/{table}/' +
                    season.split('/')[-1])
                year = self.season
            else:
                url = (
                    'https://fbref.com/en/comps/9/' +
                    season.split('/')[-2] +
                    '/{table}/' +
                    season.split('/')[-1])
                year = season.split('/')[-1][:4]

            # self.get_team_data(url).to_csv(
            #     os.path.join(self.root, f'{year}_teams.csv'),
            #     index=False)

            self.get_keeper_data(url).to_csv(
                os.path.join(self.root, f'{year}_keeper.csv'),
                index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as stat:
        season_data = json.load(stat)

    fbref = FBRef(logger, season_data)
    # fbref.get_pl_season(history=True)
    fbref.get_pl_season(history=False)
