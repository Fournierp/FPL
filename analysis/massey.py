import pandas as pd
import numpy as np

class Massey:
    """ Massey sport ranking """

    def __init__(self, fixtures, league_table, decay=False):
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

        self.results["date"] = pd.to_datetime(self.results["kickoff_time"])
        self.results["days_since"] = (self.results["date"].max() - self.results["date"]).dt.days
        self.results["weight"] = self._time_decay(0.005, self.results["days_since"]) if decay else 1

        # Get current season results
        self.home_results = self.results.copy()
        self.home_results['gd'] = self.home_results.apply(
            lambda row: self._score_delta_at_home(row), axis=1)
        self.home_results = (self.home_results.loc[:, ['team_a', 'team_h', 'gd', "weight"]])
        
        self.away_results = self.results.copy()
        self.away_results['gd'] = self.away_results.apply(
            lambda row: self._score_delta_at_away(row), axis=1)
        self.away_results = (self.away_results.loc[:, ['team_a', 'team_h', 'gd', "weight"]])

    def _score_delta_at_home(self, row):
        """
        Computes the points difference for the home team

        Args:
            row (arraw): Fixture
        """
        return row['team_h_score'] - row['team_a_score']

    def _score_delta_at_away(self, row):
        """
        Computes the points difference for the away team

        Args:
            row (arraw): Fixture
        """
        return row['team_a_score'] - row['team_h_score']

    def _aggregate_gd(self, team):
        """
        Computes the overall points accumulated teams

        Args:
            team (int): Team index
        """
        return (
            sum(
                self.home_results[self.home_results['team_h'] == team]['gd'] *
                self.home_results[self.home_results['team_h'] == team]['weight']) +
            sum(
                self.away_results[self.away_results['team_a'] == team]['gd'] *
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
        """ Computes massey ratings

        Returns:
            (pd.DataFrame): League table of ratings
        """
        lt = (self.league_table.id.map(self._aggregate_gd))
        self.league_table['gd'] = lt

        x = pd.DataFrame(
            index=self.league_table["id"],
            columns=self.league_table["id"]).fillna(0)

        for i in range(1, 21):
            x.loc[i, i] = 0

        for _, row in self.results.iterrows():
            res = 1

            x.loc[int(row['team_h']), int(row['team_h'])] += res
            x.loc[int(row['team_h']), int(row['team_a'])] = -res
            
            x.loc[int(row['team_a']), int(row['team_a'])] += res
            x.loc[int(row['team_a']), int(row['team_h'])] = -res

        X = x.values

        y = self.league_table['gd'].values

        X[19, :] = 1
        y[19] = 0

        # Solve y = Xr
        self.league_table['massey'] = np.linalg.inv(X) @ y
        return self.league_table
