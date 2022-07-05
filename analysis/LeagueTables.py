import pandas as pd
import numpy as np

from itertools import combinations

class LeagueTables:
    """ League Tables """

    def __init__(self, fixtures, league_table):
        """
        Args:
            fixtures (pd.DataFrame): Games from season to used
            league_table (pd.DataFrame): Teams present in the league
        """
        self.results = (
            fixtures[fixtures['finished'] == True]
            [['team_a', 'team_a_score', 'team_h', 'team_h_score', "event"]])
        self.league_table = league_table
        self.league_table['points'] = 0

        # Get current season results
        self.home_results = self.results.copy()
        self.home_results['points'] = self.home_results.apply(
            lambda row: self._points_at_home(row), axis=1)
        self.home_results = (self.home_results.loc[:, ['team_a', 'team_h', 'points', 'event']])
        
        self.away_results = self.results.copy()
        self.away_results['points'] = self.away_results.apply(
            lambda row: self._points_at_away(row), axis=1)
        self.away_results = (self.away_results.loc[:, ['team_a', 'team_h', 'points', 'event']])

    def _points_at_home(self, row):
        """
        Computes the points accumulated by the home team

        Args:
            row (arraw): Fixture
        """
        if row['team_a_score'] == row['team_h_score']:
            return 1
        elif row['team_a_score'] < row['team_h_score']:
            return 3
        else:
            return 0

    def _points_at_away(self, row):
        """
        Computes the points accumulated by the away team

        Args:
            row (arraw): Fixture
        """
        if row['team_a_score'] == row['team_h_score']:
            return 1
        elif row['team_a_score'] > row['team_h_score']:
            return 3
        else:
            return 0

    def _aggregate_points(self, team):
        """
        Computes the points accumulated

        Args:
            row (arraw): Fixture
        """
        return (
            self.home_results[self.home_results['team_h'] == team]['points'].sum() +
            self.away_results[self.away_results['team_a'] == team]['points'].sum())

    def lt(self):
        """
        Computes the points accumulated over a season
        """
        lt = (self.league_table.id.map(self._aggregate_points))
        self.league_table['points'] = lt
    
    def _points_per_game_difference(self, teams):
        """
        Computes the points  per game difference

        Args:
            teams (string): Team name
        """
        same_fixture = (
            self.all_results[self.all_results.team == teams[0]]
            .merge(self.all_results[self.all_results.team == teams[1]],
                how='inner',
                on=['opponent','venue']))

        if not same_fixture.empty:
            ppg_difference = (
                (same_fixture.points_x.sum() -
                same_fixture.points_y.sum())
                / same_fixture.shape[0]
                )
        else:
            ppg_difference = 0
        
        return ppg_difference

    def salt(self):
        """
        Computes the schedule adjusted league table
        """
        self.home_results = (
            self.home_results
               .assign(venue = 'home')
               .rename(columns = {
                   'team_h': 'team',
                   'team_a': 'opponent'
               })
               .loc[:, ['team', 'opponent', 'venue', 'points']])

        self.away_results = (
            self.away_results
               .assign(venue = 'away')
               .rename(columns = {
                   'team_a': 'team',
                   'team_h': 'opponent'
               })
               .loc[:, ['team', 'opponent', 'venue', 'points']])
        
        self.all_results = pd.concat([
            self.home_results, self.away_results], ignore_index=True)

        w = pd.DataFrame(
            index=list(combinations(self.league_table["id"], 2)),
            columns=self.league_table["id"]).fillna(0)

        for idx, row in w.iterrows():
            w.loc[[idx], idx[0]] = 1
            w.loc[[idx], idx[1]] = -1

        W = w.values

        r = w.index.map(self._points_per_game_difference).values.reshape(w.shape[0], 1)

        # Solve Wx = r using Least Squared Error
        x = np.linalg.lstsq(W, r, rcond=None)[0]

        mapping_table = pd.DataFrame(
            index=list(combinations(self.league_table["name"], 2)),
            columns=self.league_table["name"]).fillna(0)

        m = (self.all_results
            .groupby('team')
            .points
            .count()
            .values
            .reshape(len(mapping_table.columns), 1))

        self.league_table['adj_points'] = m * (x + self.all_results.points.mean())

    def get_gw_league_table(self, gw):
        if gw >= 0:
            home_results = self.home_results[self.home_results['event'] <= gw].copy()
            away_results = self.away_results[self.away_results['event'] <= gw].copy()

        def __aggregate_points(team):
            """
            Computes the points accumulated over a part of a season

            Args:
                team (string): Team name
            """
            return (
                home_results[home_results['team_h'] == team]['points'].sum() +
                away_results[away_results['team_a'] == team]['points'].sum())

        lt = self.league_table.copy()
        lt['points'] = (lt.id.map(__aggregate_points))
        return lt.sort_values(by=['points'], ascending=False)

    def per_gw_lt(self):
        """
        Computes the expected points league table
        """
        season_table = pd.DataFrame()
        current_gw = max(self.results['event'])

        for i in range(0, min(18, current_gw)):
            lt = self.get_gw_league_table(i)
            season_table[f'{i}'] = lt.loc[:, 'short_name'].values

        return season_table

    def _xg_vs_xga(self, row):
        """
        Computes the expected points accumulated

        Args:
            row (arraw): Fixture
        """
        if np.round(row['npxG'], 2) ==  np.round(row['npxGA'], 2):
            return 1
        elif row['npxG'] > row['npxGA']:
            return 3
        else:
            return 0

    def xlt(self):
        """
        Computes the expected points accumulated over a season
        """
        self.league_table.loc[:, 'xPoints'] = [
            (pd.read_csv('../data/understat/2021-22/Arsenal.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Aston_Villa.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Brentford.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Brighton.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Burnley.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Chelsea.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Crystal_Palace.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Everton.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Leeds.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Leicester.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Liverpool.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Manchester_City.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Manchester_United.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Newcastle_United.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Norwich.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Southampton.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Tottenham.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Watford.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/West_Ham.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum()),
            (pd.read_csv('../data/understat/2021-22/Wolverhampton_Wanderers.csv')
            .apply(lambda row: self._xg_vs_xga(row), axis=1).sum())
            ]
