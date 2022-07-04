import pandas as pd
import numpy as np

class Colley:
    """ Colley sport ranking """

    def __init__(self, fixtures, league_table, decay=False, draw_weight=.5):
        """
        Args:
            fixtures (pd.DataFrame): Games from season to used
            league_table (pd.DataFrame): Teams present in the league
            decay (boolean): Apply time decay
            draw_weight (float): Weight applied to draws
        """
        self.results = (
            fixtures[fixtures['finished'] == True]
            [['team_a', 'team_a_score', 'team_h', 'team_h_score', "kickoff_time"]])
        self.league_table = league_table
        self.draw_weight = draw_weight

        self.results["date"] = pd.to_datetime(self.results["kickoff_time"])
        self.results["days_since"] = (self.results["date"].max() - self.results["date"]).dt.days
        self.results["weight"] = self._time_decay(0.005, self.results["days_since"]) if decay else 1

        # Get current season results
        self.home_results = self.results.copy()
        self.home_results['res'] = self.home_results.apply(
            lambda row: self._score_to_points_at_home(row), axis=1)
        self.home_results = (self.home_results.loc[:, ['team_a', 'team_h', 'res', "weight"]])
        
        self.away_results = self.results.copy()
        self.away_results['res'] = self.away_results.apply(
            lambda row: self._score_to_points_at_away(row), axis=1)
        self.away_results = (self.away_results.loc[:, ['team_a', 'team_h', 'res', "weight"]])

    def _score_to_points_at_home(self, row):
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

    def _score_to_points_at_away(self, row):
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

    def _aggregate_res(self, team):
        """
        Computes the overall points accumulated teams
        Args:
            team (int): Team index
        """
        return (
            sum(
                (self.home_results[self.home_results['team_h'] == team]['res'] == 3) *
                self.home_results[self.home_results['team_h'] == team]['weight']) +
            sum(
                (self.away_results[self.away_results['team_a'] == team]['res'] == 3) *
                self.away_results[self.away_results['team_a'] == team]['weight']) -
            sum(
                (self.home_results[self.home_results['team_h'] == team]['res'] == 0) *
                self.home_results[self.home_results['team_h'] == team]['weight']) -
            sum(
                (self.away_results[self.away_results['team_a'] == team]['res'] == 0) *
                self.away_results[self.away_results['team_a'] == team]['weight'])
            )

    def _time_decay(self, xi, t):
        """ Compute importance weight based on time elapsed

        Args:
            xi (float): Decay rate
            t (int): Days elapsed

        Returns:
            (float): importance weight
        """
        return np.exp(-xi * t)

    def rating(self):
        """ Computes colley ratings

        Returns:
            (pd.DataFrame): League table of ratings
        """
        lt = (self.league_table.id.map(self._aggregate_res))
        self.league_table['res'] = lt

        x = pd.DataFrame(
            index=self.league_table["id"],
            columns=self.league_table["id"]).fillna(0)

        for i in range(1, 21):
            x.loc[i, i] = 2

        for _, row in self.results.iterrows():
            res = 1 if row['team_a_score'] != row['team_h_score'] else self.draw_weight

            x.loc[int(row['team_h']), int(row['team_h'])] += res * row['weight']
            x.loc[int(row['team_h']), int(row['team_a'])] -= res * row['weight']

            x.loc[int(row['team_a']), int(row['team_a'])] += res * row['weight']
            x.loc[int(row['team_a']), int(row['team_h'])] -= res * row['weight']

        X = x.values

        y = self.league_table['res'].values * 0.5 + 1

        # Solve y = Xr
        self.league_table['colley'] = np.linalg.inv(X) @ y
        return self.league_table
