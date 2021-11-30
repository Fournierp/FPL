import os
import requests
import logging
import time

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

# Player Standard Stats
player_stats = [
    "nationality", "position", "squad", "age", "birth_year", "games",
    "games_starts", "minutes", "goals", "assists", "goals_pens", "pens_made",
    "pens_att", "cards_yellow", "cards_red", "goals_per90", "assists_per90",
    "goals_assists_per90", "goals_pens_per90", "goals_assists_pens_per90",
    "xg", "npxg", "xa", "xg_per90", "xa_per90", "xg_xa_per90", "npxg_per90",
    "npxg_xa_per90"]

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
    "pens_made", "pens_att"]

# Player Shooting
player_shooting = [
    "shots_total", "shots_on_target", "shots_on_target_pct",
    "shots_total_per90", "shots_on_target_per90", "goals_per_shot",
    "goals_per_shot_on_target", "average_shot_distance", "shots_free_kicks",
    "pens_made", "pens_att", "xg", "npxg", "npxg_per_shot", "xg_net",
    "npxg_net"]

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
    "sca_shots", "sca_fouled", "sca_defense", "gca", "gca_per90",
    "gca_passes_live", "gca_passes_dead", "gca_dribbles", "gca_shots",
    "gca_fouled", "gca_defense"]

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

# Game Summary
summary = [
    "position", "age", "minutes", "goals", "assists", "pens_made", "pens_att",
    "shots_total", "shots_on_target", "cards_yellow", "cards_red", "touches", "pressures",
    "tackles", "interceptions", "blocks", "xg", "npxg", "xa", "sca", "gca", "passes_completed",
    "passes", "passes_pct", "progressive_passes", "carries", "progressive_carries",
    "dribbles_completed", "dribbles"]

# Game Goalkeeping
game_keepers = [
    "shots_on_target_against", "goals_against_gk", "saves", "save_pct", "psxg_gk",
    "passes_completed_launched_gk", "passes_launched_gk", "passes_pct_launched_gk",
    "passes_gk", "passes_throws_gk", "pct_passes_launched_gk", "passes_length_avg_gk",
    "goal_kicks", "pct_goal_kicks_launched", "goal_kick_length_avg", "crosses_gk",
    "crosses_stopped_gk", "crosses_stopped_pct_gk", "def_actions_outside_pen_area_gk",
    "avg_distance_def_actions_gk"]

# Game shots
game_shots = [
    "minute", "squad", "outcome", "distance", "body_part", "notes",
    "sca_1_player", "sca_1_type", "sca_2_player", "sca_2_type"]

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
        attempts = 3
        while attempts:
            try:
                res = requests.get(url)

                if res.status_code != 200:
                    raise Exception('Bad request response.')

                # Handle hidden table.
                comm = re.compile("<!--|-->")
                soup = BeautifulSoup(comm.sub("", res.text), 'lxml')
                return soup.findAll("tbody")

            except:
                attempts -= 1
                if not attempts:
                    self.logger.warning(
                        f"URL Request to {url} failed after 3 attempts.")
                    return None

                self.logger.warning(
                    f'URL Request failed, retrying in 3 seconds! URL: {url}')
                time.sleep(30)

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

        df = pd.concat(df, axis=1)
        return df.loc[:, ~df.columns.duplicated()]

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

    def get_player_data(self, url):
        """ Scrape each table of Outfield player data

        Args:
            url (string): Base link to the team stats

        Returns:
            (pd.DataFrame): Final data
        """
        categories_name = [
            'stats', 'shooting', 'passing', 'passing_types',
            'gca', 'defense', 'possession', 'misc'
            ]
        categories_cols = [
            player_stats, player_shooting, passing, passing_types,
            gca, defense, possession, misc
            ]
        df = []
        for name, cols in zip(categories_name, categories_cols):
            df.append(self.get_player_table(url.format(table=name), cols))

        df = pd.concat(df, axis=1)
        return df.loc[:, ~df.columns.duplicated()]

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

    def get_pl_season(self, history=False):
        """ Scrape data fron seasons

        Args:
            history (bool, optional): Scrape current. Defaults to False.
        """
        seasons = self.get_competition_urls(
            'https://fbref.com/en/comps/9/history/Premier-League-Seasons')

        if not history:
            seasons = [seasons[0]]

        self.logger.info("Downloading Historical Season Data")

        for season in seasons:
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

            if int(year) > 2016:
                self.logger.info(f"Season: {season}")

                df = self.get_team_data(url)
                df.loc[:, "season"] = year
                if os.path.isfile(os.path.join(self.root, 'teams.csv')):
                    past_df = pd.read_csv(os.path.join(self.root, 'teams.csv'))
                    pd.concat(
                        [past_df, df], ignore_index=True
                    ).to_csv(os.path.join(self.root, 'teams.csv'), index=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, 'teams.csv'),
                        index=False)

                df = self.get_keeper_data(url)
                df.loc[:, "season"] = year
                if os.path.isfile(os.path.join(self.root, 'keeper.csv')):
                    past_df = pd.read_csv(os.path.join(self.root, 'keeper.csv'))
                    pd.concat(
                        [past_df, df], ignore_index=True
                    ).to_csv(os.path.join(self.root, 'keeper.csv'), index=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, 'keeper.csv'),
                        index=False)

                df = self.get_player_data(url)
                df.loc[:, "season"] = year
                if os.path.isfile(os.path.join(self.root, 'outfield.csv')):
                    past_df = pd.read_csv(os.path.join(self.root, 'outfield.csv'))
                    pd.concat(
                        [past_df, df], ignore_index=True
                    ).to_csv(os.path.join(self.root, 'outfield.csv'), index=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, 'outfield.csv'),
                        index=False)

    def get_match_data_players(self, page, categories_cols):
        df = pd.DataFrame()

        for h in [1, 0]:
            df_tables = []

            for i, columns in enumerate(categories_cols):
                table_rows = page[i + (7 if not h else 0)].find_all('tr')
                match_dict = dict()

                for row in table_rows:
                    if row.find('th', {"scope": "row"}) is not None:
                        player_name = (
                            row.find('th', {"data-stat": "player"})
                            .text.strip().encode().decode("utf-8"))

                        # Add player name
                        if 'player' in match_dict:
                            match_dict['player'].append(player_name)
                        else:
                            match_dict['player'] = [player_name]

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

                                if ((col != 'position') & (col != 'age')):
                                    text = float(text.replace(',', ''))

                            if col in match_dict:
                                match_dict[col].append(text)
                            else:
                                match_dict[col] = [text]

                df_tables.append(pd.DataFrame.from_dict(match_dict))

            df = df.append(pd.concat(df_tables, axis=1))
        return df.loc[:, ~df.columns.duplicated()]

    def get_match_data_keepers(self, page, columns):
        df = pd.DataFrame()

        for h in [1, 0]:
            df_tables = []
            table_rows = page[(6 if h else 13)].find_all('tr')
            match_dict = dict()

            for row in table_rows:
                if row.find('th', {"scope": "row"}) is not None:
                    player_name = (
                        row.find('th', {"data-stat": "player"})
                        .text.strip().encode().decode("utf-8"))

                    # Add player name
                    if 'player' in match_dict:
                        match_dict['player'].append(player_name)
                    else:
                        match_dict['player'] = [player_name]

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

                            if ((col != 'position') & (col != 'age')):
                                text = float(text.replace(',', ''))

                        if col in match_dict:
                            match_dict[col].append(text)
                        else:
                            match_dict[col] = [text]

            df = df.append(pd.DataFrame.from_dict(match_dict))

        return df.loc[:, ~df.columns.duplicated()]

    def get_match_data_shots(self, page, columns):
        table_rows = page[14].find_all('tr')
        match_dict = dict()

        for row in table_rows:
            if (
                    row.find('th', {"scope": "row"}) is not None
                    and row.find('th', {"scope": "row"}).text
                    ):
                player_name = (
                    row.find('td', {"data-stat": "player"})
                    .text.strip().encode().decode("utf-8"))

                # Add player name
                if 'player' in match_dict:
                    match_dict['player'].append(player_name)
                else:
                    match_dict['player'] = [player_name]

                # Parse the table
                for col in columns:
                    # Get the statistic
                    cell = row.find("td", {"data-stat": col})
                    if cell is None:
                        # Fill na
                        text = 'None'

                        if col == 'minute':
                            text = row.find("th", {"data-stat": "minute"}).text
                            text = float(text.replace('+', '.'))

                    else:
                        a = cell.text.strip().encode()
                        text = a.decode("utf-8")
                        # Fill na
                        if(text == ''):
                            text = '0'

                    if col in match_dict:
                        match_dict[col].append(text)
                    else:
                        match_dict[col] = [text]

        df = pd.DataFrame.from_dict(match_dict)
        return df.loc[:, ~df.columns.duplicated()]

    def get_pl_season_games(self, history=False):
        seasons = self.get_competition_urls(
            'https://fbref.com/en/comps/9/history/Premier-League-Seasons')

        if not history:
            seasons = [seasons[0]]
        else:
            seasons = seasons[1:]

        self.logger.info("Downloading Historical Season Data")

        for season in seasons:
            self.logger.info(f"Season: {season}")
            if season.split('/')[-2] == '9':
                url = (
                    'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures')
                year = self.season

            else:
                url = (
                    'https://fbref.com/en/comps/9/' +
                    season.split('/')[-2] +
                    '/schedule/' +
                    season.split('/')[-1][:-6] +
                    '-Scores-and-Fixtures')
                year = season.split('/')[-1][:4]

            # Skip years with no underlying stats
            if int(year) > 2016:
                table_rows = self.get_url(url)[0].find_all('tr')

                for row in table_rows:
                    # Skip blank rows, and postponed games
                    if (
                            row.find('th', {"scope": "row"}) is not None
                            and row.find('td', {"data-stat": "match_report"}).text != ""
                            ):

                        # Skip upcoming games
                        if 'stathead' in row.find('td', {"data-stat": "match_report"}).find('a')['href']:
                            continue

                        date = row.find('td', {"data-stat": "date"}).text
                        squad_h = row.find('td', {"data-stat": "squad_a"}).text
                        squad_a = row.find('td', {"data-stat": "squad_b"}).text
                        att = row.find('td', {"data-stat": "attendance"}).text.replace(',', '')

                        df = pd.DataFrame.from_records(
                            [
                                {
                                    'date': date,
                                    'time': row.find('td', {"data-stat": "time"}).text,
                                    'squad_h': squad_h,
                                    'squad_a': squad_a,
                                    'stadium': row.find('td', {"data-stat": "venue"}).text,
                                    'referee': row.find('td', {"data-stat": "referee"}).text,
                                    'attendance': (
                                        0 if att == '' else int(att)
                                        )
                                }
                            ]
                        )

                        if os.path.isfile(os.path.join(self.root, f'games.csv')):
                            past_df = pd.read_csv(os.path.join(self.root, f'games.csv'))
                            pd.concat(
                                [past_df, df], ignore_index=True
                            ).to_csv(os.path.join(self.root, f'games.csv'), index=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games.csv'),
                                index=False)

                        page = self.get_url(
                            "https://fbref.com" +
                            row.find('td', {"data-stat": "match_report"}).find('a')['href']
                            )

                        df = self.get_match_data_players(
                            page,
                            [summary, passing, passing_types, defense, possession, misc]
                            )
                        df['date'] = date
                        df['squad_h'] = squad_h
                        df['squad_a'] = squad_a

                        if os.path.isfile(os.path.join(self.root, f'games_players.csv')):
                            past_df = pd.read_csv(os.path.join(self.root, f'games_players.csv'))
                            pd.concat(
                                [past_df, df], ignore_index=True
                            ).to_csv(os.path.join(self.root, f'games_players.csv'), index=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_players.csv'),
                                index=False)

                        df = self.get_match_data_keepers(
                            page,
                            game_keepers)
                        df['date'] = date
                        df['squad_h'] = squad_h
                        df['squad_a'] = squad_a

                        if os.path.isfile(os.path.join(self.root, f'games_keepers.csv')):
                            past_df = pd.read_csv(os.path.join(self.root, f'games_keepers.csv'))
                            pd.concat(
                                [past_df, df], ignore_index=True
                            ).to_csv(os.path.join(self.root, f'games_keepers.csv'), index=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_keepers.csv'),
                                index=False)

                        df = self.get_match_data_shots(
                            page,
                            game_shots
                            )
                        df['date'] = date
                        df['squad_h'] = squad_h
                        df['squad_a'] = squad_a

                        if os.path.isfile(os.path.join(self.root, f'games_shots.csv')):
                            past_df = pd.read_csv(os.path.join(self.root, f'games_shots.csv'))
                            pd.concat(
                                [past_df, df], ignore_index=True
                            ).to_csv(os.path.join(self.root, f'games_shots.csv'), index=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_shots.csv'),
                                index=False)

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
    # fbref.get_pl_season(history=True)
    # fbref.get_pl_season(history=False)
    fbref.get_pl_season_games(history=True)
    fbref.get_pl_season_games(history=False)
    # fbref.get_fixtures(True)
