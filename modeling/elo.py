import pandas as pd
import numpy as np
import json

from utils import get_next_gw
from ranked_probability_score import ranked_probability_score, match_outcome


class Elo:

    def __init__(self, games):
        teams = np.sort(np.unique(games["team1"]))
        league_size = len(teams)

        self.teams = (
            games.loc[:, ["team1"]]
            .drop_duplicates()
            .sort_values("team1")
            .reset_index(drop=True)
            .assign(team_index=np.arange(league_size))
            .assign(rating=1500)
            .rename(columns={"team1": "team"})
        )
        self.league_size = self.teams.shape[0]

        df = (
            pd.merge(games, self.teams, left_on="team1", right_on="team")
            .rename(columns={"team_index": "hg"})
            .drop(["team"], axis=1)
            .drop_duplicates()
            .merge(self.teams, left_on="team2", right_on="team")
            .rename(columns={"team_index": "ag"})
            .drop(["team"], axis=1)
        )

        self.games = df.loc[:, ["score1", "score2", "team1", "team2", "hg", "ag", "date"]]
        self.games["winner"] = match_outcome(self.games)
        self.games["date"] = pd.to_datetime(self.games["date"], dayfirst=True)# format='%d/%m/%y')
        self.games = self.games.sort_values("date")

        self.historical_rating = self.games.loc[:, ["team1", "team2", "hg", "ag", "date"]]


    def fit(self, k=20, w=400, hfa=50):

        def predict(home_rating, away_rating):
            return 1 / (1 + pow(10, (away_rating - home_rating) / w))


        def rating_update(rating, actual_score, expected_score):
            return rating + k * (actual_score - expected_score)


        for index, match in self.games.iterrows():
            # Get match data
            home_team = match['team1']
            away_team = match['team2']

            home_rating = self.teams.loc[self.teams.team == home_team]['rating'].values[0]
            away_rating = self.teams.loc[self.teams.team == away_team]['rating'].values[0]

            # Save the rating prior to the game
            self.historical_rating.loc[
                (
                    (self.historical_rating.team1 == home_team) &
                    (self.historical_rating.team2 == away_team) &
                    (self.historical_rating.date == match['date'])),
                'rating1'] = home_rating
            self.historical_rating.loc[
                (
                    (self.historical_rating.team1 == home_team) &
                    (self.historical_rating.team2 == away_team) &
                    (self.historical_rating.date == match['date'])),
                'rating2'] = away_rating

            # Infer result
            exp_h = predict(home_rating + hfa, away_rating)
            exp_a = predict(away_rating, home_rating + hfa)

            # Update ratings
            res_h = 1 if match['winner'] == 0 else 0.5 if match['winner'] == 1 else 0
            res_a = 1 if match['winner'] == 2 else 0.5 if match['winner'] == 1 else 0

            self.teams.loc[self.teams.team == home_team, 'rating'] = rating_update(home_rating, res_h, exp_h)
            self.teams.loc[self.teams.team == away_team, 'rating'] = rating_update(away_rating, res_a, exp_a)


if __name__ == "__main__":
    games = (
        pd.read_csv(
            f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
            usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
            
            for season in [
                '9394', '9495', '9596', '9697', '9798',
                '9899', '9900', '0001', '0102', '0203',
                '0304', '0405', '0506', '0607', '0708',
                '0809', '0910', '1011', '1112', '1213',
                '1314', '1415', '1516', '1617', '1718',
                '1819', '1920', '2021', '2122'])

    games = pd.concat(games).rename(columns={
        "HomeTeam": "team1",
        "AwayTeam": "team2",
        "FTHG": "score1",
        "FTAG": "score2",
        "Date": "date",
        }).dropna()

    # Train model on all games up to the previous GW
    model = Elo(games)
    model.fit()

    model.teams.to_csv('team.csv')
    model.historical_rating.to_csv('historical_rating.csv')
