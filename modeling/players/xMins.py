import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.linear_model import LinearRegression


class xMinutes:

    def __init__(self, games):
        self.games = games.sort_values(by=['Date']).reset_index(drop=True)

    def uniform(self):
        _, _, _, y_test = train_test_split(self.games, 'Date', 'Min')
        y_pred = y_test.copy()
        y_pred = np.random.uniform(0, 90)

        return y_test, y_pred

    def constant(self):
        _, _, _, y_test = train_test_split(self.games, 'Date', 'Min')
        y_pred = y_test.copy()
        y_pred = 45

        return y_test, y_pred

    def evaluate(self, y_test, y_pred):
        return np.sqrt(np.mean(np.power(y_test - y_pred, 2)))

    def last_game_minutes_lagged_linear_reg(self):
        self.games[f"Min_minus_1"] = self.games['Min'].shift(1)

        self.games = self.games.iloc[1:]

        X_train, X_test, y_train, y_test = train_test_split(self.games, ['Min_minus_1'], 'Min')
        reg = LinearRegression().fit(X_train, y_train)

        return y_test, reg.predict(X_test)

    def minutes_lagged_linear_reg(self):
        lag_values = list(range(1, 6 + 1))

        features_lags = []
        lagged_column = 'Min'

        for lag_value in lag_values:
            self.games[f"{lagged_column}_minus_{lag_value}"] = self.games[lagged_column].shift(lag_value)
            features_lags.append(f"{lagged_column}_minus_{lag_value}")

        self.games = self.games.iloc[6:]

        X_train, X_test, y_train, y_test = train_test_split(self.games, features_lags, 'Min')

        reg = LinearRegression().fit(X_train, y_train)

        return y_test, reg.predict(X_test)

    def rolling_minutes_linear_reg(self):
        self.games[f"Min_minus_1_to_6"] = self.games['Min'].shift(1).rolling(6).mean()

        self.games = self.games.iloc[6:]

        X_train, X_test, y_train, y_test = train_test_split(self.games, ['Min_minus_1_to_6'], 'Min')

        reg = LinearRegression().fit(X_train, y_train)

        return y_test, reg.predict(X_test)

    def weighted_rolling_minutes_linear_reg(self):
        weights = np.array([.03, .085, .14, .195, .2475, .3025])
        self.games[f"Min_minus_1_to_6"] = self.games['Min'].shift(1).rolling(6).apply(lambda x: np.sum(weights*x))

        self.games = self.games.iloc[6:]

        X_train, X_test, y_train, y_test = train_test_split(self.games, ['Min_minus_1_to_6'], 'Min')

        reg = LinearRegression().fit(X_train, y_train)

        return y_test, reg.predict(X_test)

    def weighted_rolling_cases_linear_reg(self):
        weights = np.array([.03, .085, .14, .195, .2475, .3025])

        cases_lags = []

        for lagged_column in ['A', 'B', 'C', 'D']:
            self.games[f"{lagged_column}_minus_1_to_6"] = self.games[lagged_column].shift(1).rolling(6).apply(lambda x: np.sum(weights*x))
            cases_lags.append(f"{lagged_column}_minus_1_to_6")

        self.games = self.games.iloc[6:]

        X_train, X_test, y_train, y_test = train_test_split(self.games, cases_lags, 'Min')

        reg = LinearRegression().fit(X_train, y_train)

        return y_test, reg.predict(X_test)

    def linear_reg(self):
        weights = np.array([.03, .085, .14, .195, .2475, .3025])

        features_lags, cases_lags = [], []
        lagged_column = 'Min'

        for lag_value in range(1, 7):
            self.games[f"{lagged_column}_minus_{lag_value}"] = self.games[lagged_column].shift(lag_value)
            features_lags.append(f"{lagged_column}_minus_{lag_value}")

        self.games[f"Min_minus_1_to_6"] = self.games['Min'].shift(1).rolling(6).apply(lambda x: np.sum(weights*x))

        for lagged_column in ['A', 'B', 'C', 'D']:
            self.games[f"{lagged_column}_minus_1_to_6"] = self.games[lagged_column].shift(1).rolling(6).apply(lambda x: np.sum(weights*x))
            cases_lags.append(f"{lagged_column}_minus_1_to_6")

        self.games = self.games.iloc[6:]

        X_train, X_test, y_train, y_test = train_test_split(self.games, ['Min_minus_1_to_6'] + cases_lags + features_lags, 'Min')

        reg = LinearRegression().fit(X_train, y_train)

        return y_test, reg.predict(X_test)

def train_test_split(df, covariates, target):
    X_train = df.iloc[:-15][covariates].reset_index(drop=True)
    y_train = df.iloc[:-15][target].reset_index(drop=True)
    X_test = df.iloc[-15:][covariates].reset_index(drop=True)
    y_test = df.iloc[-15:][target].reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def player_match_history(name, club):
    # TODO: Handle newly transfered players with match history of games he was not here for
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
    all_fixtures['C'] = np.where(all_fixtures.Min == 0, 1, 0)
    all_fixtures['D'] = np.where(all_fixtures['Subbed on'] == 1, 1, 0)

    # Keep fixtures that were played
    return all_fixtures.loc[all_fixtures.Date < datetime.today()]


if __name__ == "__main__":
    games = player_match_history('Kevin De Bruyne', 'Manchester City')

    xMins = xMinutes(games)

    predictions = xMins.evaluate(*xMins.uniform())
    print(f">>> Uniform baseline: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.constant())
    print(f">>> Constant baseline: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.last_game_minutes_lagged_linear_reg())
    print(f">>> Last game minutes LR: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.minutes_lagged_linear_reg())
    print(f">>> Last 6 game minutes LR: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.rolling_minutes_linear_reg())
    print(f">>> Last 6 game rolling minutes LR: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.weighted_rolling_minutes_linear_reg())
    print(f">>> Last 6 game weighted rolling minutes LR: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.weighted_rolling_cases_linear_reg())
    print(f">>> Last 6 game weighted rolling cases LR: {predictions:.2f}")

    predictions = xMins.evaluate(*xMins.linear_reg())
    print(f">>> LR: {predictions:.2f}")
