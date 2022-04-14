import pandas as pd
import numpy as np

from datetime import datetime


class xMins:

    def __init__(self, games):
        self.games = games

    def uniform(self):
        self.games["xmins"] = np.random.uniform(0, 90)

        return self.games

    def evaluate(self, games):
        games["error"] = games.apply(
            lambda row: np.power(row.xmins - row.Min, 2),
            axis=1)

        return np.sqrt(np.mean(games.error))

def player_match_history(name, club):
    # Player minutes
    lineup_df = pd.read_csv('data/fbref/games_lineup.csv')
    lineup_df = lineup_df.loc[lineup_df.Player == name].loc[(lineup_df.squad_h == club) | (lineup_df.squad_a == club)]
    lineup_df["date"] = pd.to_datetime(lineup_df["date"])

    # Historical PL Fixtures
    fixtures_df = pd.read_csv("data/fbref/fixtures.csv")
    fixtures_df = fixtures_df.loc[fixtures_df.Competition == 'Premier-League']
    fixtures_df = fixtures_df[["Date", "Home", "Away"]]
    fixtures_df["Date"] = pd.to_datetime(fixtures_df["Date"])

    # Add data when the player is not included in the team
    # And keep only fixtures of the current team
    all_fixtures = fixtures_df.loc[(fixtures_df.Home == club) | (fixtures_df.Away == club)]
    all_fixtures = pd.merge(
        lineup_df,
        all_fixtures,
        left_on=['date', 'squad_h', 'squad_a'],
        right_on=['Date', 'Home', 'Away'],
        how='right'
    )
    all_fixtures.Player = name
    all_fixtures = all_fixtures.fillna(0)
    all_fixtures.home = np.where(all_fixtures.Home == club, 1, 0)
    all_fixtures = all_fixtures.drop(['date', 'squad_h', 'squad_a'], axis=1)
    all_fixtures['Subbed on'] = np.where((all_fixtures.Benched == 1) & (all_fixtures.Min > 0), 1, 0)
    all_fixtures['Subbed off'] = np.where((all_fixtures.Starter == 1) & (all_fixtures.Min < 90), 1, 0)

    all_fixtures['A'] = np.where(all_fixtures.Min == 90, 1, 0)
    all_fixtures['B'] = np.where(all_fixtures['Subbed off'] == 1, 1, 0)
    all_fixtures['C'] = np.where((all_fixtures.Lineup == 0) | (all_fixtures['Subbed on'] == 0), 1, 0)
    all_fixtures['D'] = np.where(all_fixtures['Subbed on'] == 1, 1, 0)

    # Keep fixtures that were played
    return all_fixtures.loc[all_fixtures.Date < datetime.today()]


if __name__ == "__main__":
    games = player_match_history('Mohamed Salah', 'Liverpool')

    xMins = xMins(games)

    predictions = xMins.evaluate(xMins.uniform())
    print(f"{predictions:.2f}")
