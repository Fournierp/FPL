import pandas as pd
import numpy as np

class Colley:
    """ """

    def __init__(self, fixtures, league_table, decay=False):
        self.results = (
            fixtures[fixtures['finished'] == True]
            [['team_a', 'team_a_score', 'team_h', 'team_h_score', "kickoff_time"]])
        self.league_table = league_table

        self.results["date"] = pd.to_datetime(self.results["kickoff_time"])
        self.results["days_since"] = (self.results["date"].max() - self.results["date"]).dt.days
        self.results["weight"] = self._time_decay(0.005, self.results["days_since"]) if decay else 1

        self.home_results = self.results.copy()
        self.home_results['res'] = self.home_results.apply(
            lambda row: self._score_to_points_at_home(row), axis=1)
        self.home_results = (self.home_results.loc[:, ['team_a', 'team_h', 'res', "weight"]])
        
        self.away_results = self.results.copy()
        self.away_results['res'] = self.away_results.apply(
            lambda row: self._score_to_points_at_away(row), axis=1)
        self.away_results = (self.away_results.loc[:, ['team_a', 'team_h', 'res', "weight"]])

    def _score_to_points_at_home(self, row):
        if row['team_a_score'] == row['team_h_score']:
            return 1
        elif row['team_a_score'] < row['team_h_score']:
            return 3
        else:
            return 0

    def _score_to_points_at_away(self, row):
        if row['team_a_score'] == row['team_h_score']:
            return 1
        elif row['team_a_score'] > row['team_h_score']:
            return 3
        else:
            return 0

    def _aggregate_res(self, team):
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
        return np.exp(-xi * t)

    def rating(self):
        lt = (self.league_table.id.map(self._aggregate_res))
        self.league_table['res'] = lt

        x = pd.DataFrame(
            index=self.league_table["id"],
            columns=self.league_table["id"]).fillna(0)

        for i in range(1, 21):
            x.loc[i, i] = 2

        for _, row in self.results.iterrows():
            x.loc[int(row['team_h']), int(row['team_h'])] += row['weight']
            x.loc[int(row['team_h']), int(row['team_a'])] += -row['weight']

            x.loc[int(row['team_a']), int(row['team_a'])] += row['weight']
            x.loc[int(row['team_a']), int(row['team_h'])] += -row['weight']

        X = x.values

        y = self.league_table['res'].values * 0.5 + 1

        # Solve y = Xr
        self.league_table['colley'] = np.linalg.inv(X) @ y
        return self.league_table.sort_values('colley')
