import os
import requests
import logging
import time
from datetime import datetime

import json
from bs4 import BeautifulSoup
import re
import pandas as pd


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

# Player Standard Stats
player_stats = [
    "nationality", "position", "squad", "age", "birth_year", "games",
    "games_starts", "minutes", "goals", "assists", "goals_pens", "pens_made",
    "pens_att", "cards_yellow", "cards_red", "goals_per90", "assists_per90",
    "goals_assists_per90", "goals_pens_per90", "goals_assists_pens_per90",
    "xg", "npxg", "xa", "xg_per90", "xa_per90", "xg_xa_per90", "npxg_per90",
    "npxg_xa_per90"]

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
                [
                    'Premier-League', 'EFL-Cup', "FA-Cup",
                    "Champions-League", "Europa-League"]):

            # Get links of historical competitions
            seasons = self.get_competition_urls(
                f'https://fbref.com/en/comps/{index}/history/{comp}-Seasons')

            self.logger.info(f"Downloading {comp} Fixtures Data")
            for season in seasons:
                if season.split('/')[-2] == index:
                    url = (
                        f'https://fbref.com/en/comps/{index}/schedule' +
                        f'/{comp}-Scores-and-Fixtures')
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

                    if "Wk" in df.columns:
                        if (
                                comp == "Champions-League" or
                                comp == "Europa-League"):
                            df = df.drop(["Wk"], axis=1)
                        else:
                            df = df.rename(columns={'Wk': "Round"})

                    df = df.loc[:, [
                        "Round", "Day", "Date", "Time", "Home", "Score",
                        "Away", "Attendance", "Venue", "Referee",
                        "Notes", "Competition"]]

                    if os.path.isfile(os.path.join(self.root, 'fixtures.csv')):
                        df.to_csv(
                            os.path.join(self.root, 'fixtures.csv'),
                            index=False, mode='a', header=False)
                    else:
                        df.to_csv(
                            os.path.join(self.root, 'fixtures.csv'),
                            index=False)

    def get_team_data(self, url):
        """ Scrape each table of team data

        Args:
            url (string): Base link to the team stats

        Returns:
            (pd.DataFrame): Final data
        """
        df, df_opp = [], []
        # URL Request
        tables = pd.read_html(url.format(table=""))
        for i, table in enumerate(tables):
            if i > 1 or i == 0 :
                if i != 0 :
                    table.columns = [' '.join(col).strip() if "Unnamed" not in col[0] else col[1] for i, col in enumerate(table.columns.values)]
                else:
                    table = table.sort_values(by=['Squad']).reset_index().drop(['Rk', 'index'], 1)

                if not i % 2:
                    df.append(table)
                else:
                    df_opp.append(table)

        df = pd.concat(df, axis=1)
        df_opp = pd.concat(df_opp, axis=1)
        return df.loc[:, ~df.columns.duplicated()], df_opp.loc[:, ~df_opp.columns.duplicated()]

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
                    f'URL Request failed, retrying in 30 seconds! URL: {url}')
                time.sleep(30)

    def get_player_table(self, url, columns):
        """ Parse the table of outfield or keeper player data

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
            df.append(self.get_player_table(url.format(table=name + "/"), cols))

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
            df.append(self.get_player_table(url.format(table=name + "/"), cols))

        df = pd.concat(df, axis=1)
        return df.loc[:, ~df.columns.duplicated()]

    def get_pl_season(self, history=False):
        """ Scrape data from seasons

        Args:
            history (bool, optional): Scrape current. Defaults to False.
        """
        seasons = self.get_competition_urls(
            'https://fbref.com/en/comps/9/history/Premier-League-Seasons')

        if not history:
            seasons = [seasons[0]]

        self.logger.info("Downloading Season Data")

        for season in seasons:
            if season.split('/')[-2] == '9':
                url = (
                    'http://fbref.com/en/comps/9/{table}' +
                    season.split('/')[-1])
                year = self.season

            else:
                url = (
                    'https://fbref.com/en/comps/9/' +
                    season.split('/')[-2] +  '/{table}' +
                    season.split('/')[-1])
                year = season.split('/')[-1][:4]

            if int(year) > 2016:
                self.logger.info(f"Season: {season}")

                df, df_opp = self.get_team_data(url)
                df.loc[:, "season"] = year
                df_opp.loc[:, "season"] = year
                if os.path.isfile(os.path.join(self.root, 'team.csv')):
                    df.to_csv(
                        os.path.join(self.root, 'team.csv'),
                        index=False, mode='a', header=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, 'team.csv'),
                        index=False)
                if os.path.isfile(os.path.join(self.root, 'team_opp.csv')):
                    df_opp.to_csv(
                        os.path.join(self.root, 'team_opp.csv'),
                        index=False, mode='a', header=False)
                else:
                    df_opp.to_csv(
                        os.path.join(self.root, 'team_opp.csv'),
                        index=False)

                df = self.get_keeper_data(url)
                df.loc[:, "season"] = year
                if os.path.isfile(os.path.join(self.root, 'keeper.csv')):
                    df.to_csv(
                        os.path.join(self.root, 'keeper.csv'),
                        index=False, mode='a', header=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, 'keeper.csv'),
                        index=False)

                df = self.get_player_data(url)
                df.loc[:, "season"] = year
                if os.path.isfile(os.path.join(self.root, 'outfield.csv')):
                    df.to_csv(
                        os.path.join(self.root, 'outfield.csv'),
                        index=False, mode='a', header=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, 'outfield.csv'),
                        index=False)

    def get_games_players(self, tables):
        """ Get data about the outfielders who played

        Args:
            tables (list): DataFrames extracted from the url of fixture

        Returns:
            pd.DataFrames: Data of the fixture
        """
        df_h = []
        df_a = []

        # Home player
        for i in range(3, 9):
            table = tables[i].copy()
            table.columns = [' '.join(col).strip() if i > 5 else col[1] for i, col in enumerate(table.columns.values)]
            table = table.dropna(subset=["Nation"])
            table.loc[:, 'home'] = 1
            df_h.append(table)

        # Away player
        for i in range(10, 16):
            table = tables[i].copy()
            table.columns = [' '.join(col).strip() if i > 5 else col[1] for i, col in enumerate(table.columns.values)]
            table = table.dropna(subset=["Nation"])
            table.loc[:, 'home'] = 0
            df_a.append(table)

        df = pd.concat(
            [pd.concat(df_h, axis=1),
            pd.concat(df_a, axis=1)])
        return df.loc[:, ~df.columns.duplicated()]

    def get_games_keepers(self, tables):
        """ Get data about the keepers who played

        Args:
            tables (list): DataFrames extracted from the url of fixture

        Returns:
            pd.DataFrames: Data of the fixture
        """
        # Home keeper
        table_h = tables[9].copy()
        table_h.columns = [' '.join(col).strip() if i > 5 else col[1] for i, col in enumerate(table_h.columns.values)]
        table_h = table_h.dropna(subset=["Nation"])
        table_h.loc[:, 'home'] = 1

        # Away keeper
        table_a = tables[16].copy()
        table_a.columns = [' '.join(col).strip() if i > 5 else col[1] for i, col in enumerate(table_a.columns.values)]
        table_a = table_a.dropna(subset=["Nation"])
        table_a.loc[:, 'home'] = 0

        df = pd.concat([table_h, table_a])
        return df.loc[:, ~df.columns.duplicated()]

    def get_games_lineups(self, tables):
        """ Get data about the players who played and got subsituted

        Args:
            tables (list): DataFrames extracted from the url of fixture

        Returns:
            pd.DataFrames: Data of the fixture
        """
        df_h = []
        df_a = []

        # Home roster
        starting_lineup_h = tables[0][[tables[0].columns[1]]]
        starting_lineup_h['Lineup'] = 1

        starting_lineup_h.loc[:11, 'Starter'] = 1
        starting_lineup_h.loc[11:, 'Benched'] = 1

        starting_lineup_h = starting_lineup_h.drop(11)
        starting_lineup_h = starting_lineup_h.rename(columns={tables[0].columns[1]: 'Player'})
        starting_lineup_h.loc[:, 'home'] = 1

        # Away roster
        starting_lineup_a = tables[1][[tables[1].columns[1]]]
        starting_lineup_a['Lineup'] = 1

        starting_lineup_a.loc[:11, 'Starter'] = 1
        starting_lineup_a.loc[11:, 'Benched'] = 1

        starting_lineup_a = starting_lineup_a.drop(11)
        starting_lineup_a = starting_lineup_a.rename(columns={tables[1].columns[1]: 'Player'})
        starting_lineup_a.loc[:, 'home'] = 0

        # Home minutes played & substitutions
        minutes_h = tables[3][[('Unnamed: 0_level_0', 'Player'), ('Unnamed: 5_level_0', 'Min')]]
        minutes_h.columns = minutes_h.columns.map(lambda x: x[1])
        minutes_h = minutes_h.iloc[:-1]

        # Away minutes played & substitutions
        minutes_a = tables[10][[('Unnamed: 0_level_0', 'Player'), ('Unnamed: 5_level_0', 'Min')]]
        minutes_a.columns = minutes_a.columns.map(lambda x: x[1])
        minutes_a = minutes_a.iloc[:-1]

        df_h = pd.merge(
            starting_lineup_h,
            minutes_h,
            how='outer',
            left_on='Player',
            right_on='Player'
        )

        df_a = pd.merge(
            starting_lineup_a,
            minutes_a,
            how='outer',
            left_on='Player',
            right_on='Player'
        )
        df = pd.concat([df_h, df_a])

        return df.fillna(0)

    def get_pl_games(self, history=False):
        """ Get every PL fixture data

        Args:
            history (boolean): Collect historical data
        """
        seasons = self.get_competition_urls(
            'https://fbref.com/en/comps/9/history/Premier-League-Seasons')

        if not history:
            seasons = [seasons[0]]

        self.logger.info("Downloading Match Data")

        for season in seasons:
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
                self.logger.info(f"Season: {season}")

                # URL Request
                df = (
                    pd.read_html(url)[0]
                    .loc[:, [
                        'Wk', 'Day', 'Date', 'Time', 'Home', 'Away',
                        'Attendance', 'Venue', 'Referee', 'Notes']]
                        )
                # Remove empty row
                df = df[~df.Wk.isna()]
                # Remove upcoming games
                df = df[df.Date < datetime.now().strftime("%Y-%m-%d")]
                # Save
                if os.path.isfile(os.path.join(self.root, f'games.csv')):
                    df.to_csv(
                        os.path.join(self.root, 'games.csv'),
                        index=False, mode='a', header=False)
                else:
                    df.to_csv(
                        os.path.join(self.root, f'games.csv'),
                        index=False)

                # Get urls to games
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

                        tables = pd.read_html(
                            "https://fbref.com" +
                            row.find('td', {"data-stat": "match_report"}).find('a')['href'])

                        df = self.get_games_players(tables)
                        df.loc[:, 'date'] = date
                        df.loc[:, 'squad_h'] = squad_h
                        df.loc[:, 'squad_a'] = squad_a

                        if os.path.isfile(os.path.join(self.root, f'games_players.csv')):
                            df.to_csv(
                                os.path.join(self.root, 'games_players.csv'),
                                index=False, mode='a', header=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_players.csv'),
                                index=False)

                        df = self.get_games_keepers(tables)
                        df.loc[:, 'date'] = date
                        df.loc[:, 'squad_h'] = squad_h
                        df.loc[:, 'squad_a'] = squad_a

                        if os.path.isfile(os.path.join(self.root, f'games_keepers.csv')):
                            df.to_csv(
                                os.path.join(self.root, 'games_keepers.csv'),
                                index=False, mode='a', header=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_keepers.csv'),
                                index=False)

                        df = tables[17].copy()
                        df.columns = [' '.join(col).strip() if i > 6 else col[1] for i, col in enumerate(df.columns.values)]
                        df = df[~df.Player.isna()]
                        df.loc[:, 'date'] = date

                        if os.path.isfile(os.path.join(self.root, f'games_shots.csv')):
                            df.to_csv(
                                os.path.join(self.root, 'games_shots.csv'),
                                index=False, mode='a', header=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_shots.csv'),
                                index=False)

                        df = self.get_games_lineups(tables)
                        df.loc[:, 'date'] = date
                        df.loc[:, 'squad_h'] = squad_h
                        df.loc[:, 'squad_a'] = squad_a

                        if os.path.isfile(os.path.join(self.root, f'games_lineup.csv')):
                            df.to_csv(
                                os.path.join(self.root, 'games_lineup.csv'),
                                index=False, mode='a', header=False)
                        else:
                            df.to_csv(
                                os.path.join(self.root, f'games_lineup.csv'),
                                index=False)

        # Drop dupplicates in case I run the latest season scraper to update it.
        (
            pd.read_csv(os.path.join(self.root, 'games.csv'))
            .drop_duplicates()
            .to_csv(os.path.join(self.root, 'games.csv'), index=False))
        
        (
            pd.read_csv(os.path.join(self.root, 'games_players.csv'))
            .drop_duplicates()
            .to_csv(os.path.join(self.root, 'games_players.csv'), index=False))
        
        (
            pd.read_csv(os.path.join(self.root, 'games_keepers.csv'))
            .drop_duplicates()
            .to_csv(os.path.join(self.root, 'games_keepers.csv'), index=False))
        
        (
            pd.read_csv(os.path.join(self.root, 'games_shots.csv'))
            .drop_duplicates()
            .to_csv(os.path.join(self.root, 'games_shots.csv'), index=False))

        (
            pd.read_csv(os.path.join(self.root, 'games_lineup.csv'))
            .drop_duplicates()
            .to_csv(os.path.join(self.root, 'games_lineup.csv'), index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    with open('info.json') as stat:
        season_data = json.load(stat)

    fbref = FBRef(logger, season_data)
    # fbref.get_fixtures()

    # fbref.get_pl_season(True)

    fbref.get_pl_games()