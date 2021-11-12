import pandas as pd
import numpy as np
import os

from utils import get_next_gw
from ranked_probability_score import ranked_probability_score, match_outcome


class Elo:
    """ Elo rating system based on the outcomes of the games """

    def __init__(self, games):
        """
        Args:
            games (pd.DataFrame): Finished games to used for training.
        """
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

        self.games = (
            pd.merge(games, self.teams, left_on="team1", right_on="team")
            .rename(columns={"team_index": "hg"})
            .drop(["team"], axis=1)
            .drop_duplicates()
            .merge(self.teams, left_on="team2", right_on="team")
            .rename(columns={"team_index": "ag"})
            .drop(["team"], axis=1)
            .loc[:, [
                "score1", "score2", "team1", "team2", "hg", "ag", "date"]])
        self.games["winner"] = match_outcome(self.games)
        self.games["date"] = pd.to_datetime(self.games["date"], dayfirst=True)
        self.games = self.games.sort_values("date")

        self.historical_rating = self.games.loc[:, [
            "team1", "team2", "hg", "ag", "date"]]

    def odds(self, rating_a, rating_b, w=400):
        """ Compute the expected winning odd of team A in a fixture

        Args:
            rating_a (int): Rating of the team
            rating_b (int): Rating of the team
            w (int, optional): width of the distribution. Defaults to 400.

        Returns:
            float: expected winning odd of team A in a fixture
        """
        return 1 / (1 + pow(10, (rating_b - rating_a) / w))

    def rating_update(self, rating, actual_score, expected_score, k=20):
        """ Amount by which the rating of a team will be changed based on the outcome
        and expectation of a game

        Args:
            rating (int): Rating of the team
            actual_score (float): Result
            expected_score (float): Expected result
            k (int, optional): factor by which a victory changes rating

        Returns:
            (float): elo points delta
        """
        return rating + k * (actual_score - expected_score)

    def fit(self, hfa=50):
        """ Compute the current team ratings based on past results

        Args:
            hfa (int, optional): Home field advantage. Defaults to 50.
        """
        if os.path.isfile("data/predictions/scores/teams.csv"):
            self.teams = pd.read_csv("data/predictions/scores/teams.csv")

        else:
            for index, match in self.games.iterrows():
                # Get match data
                home_team = match['team1']
                away_team = match['team2']

                home_rating = self.teams.loc[
                    self.teams.team == home_team]['rating'].values[0]
                away_rating = self.teams.loc[
                    self.teams.team == away_team]['rating'].values[0]

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
                exp_h = self.odds(home_rating + hfa, away_rating)
                exp_a = self.odds(away_rating, home_rating + hfa)

                # Save the probabilities
                self.historical_rating.loc[
                    (
                        (self.historical_rating.team1 == home_team) &
                        (self.historical_rating.team2 == away_team) &
                        (self.historical_rating.date == match['date'])),
                    'home_win_p'] = exp_h
                self.historical_rating.loc[
                    (
                        (self.historical_rating.team1 == home_team) &
                        (self.historical_rating.team2 == away_team) &
                        (self.historical_rating.date == match['date'])),
                    'away_win_p'] = exp_a
                self.historical_rating.loc[
                    (
                        (self.historical_rating.team1 == home_team) &
                        (self.historical_rating.team2 == away_team) &
                        (self.historical_rating.date == match['date'])),
                    'draw_p'] = 0

                # Update ratings
                res_h = (
                    1 if match['winner'] == 0
                    else 0.5 if match['winner'] == 1
                    else 0)
                res_a = (
                    1 if match['winner'] == 2
                    else 0.5 if match['winner'] == 1
                    else 0)

                self.teams.loc[
                    self.teams.team == home_team, 'rating'] = \
                    self.rating_update(home_rating, res_h, exp_h)
                self.teams.loc[
                    self.teams.team == away_team, 'rating'] = \
                    self.rating_update(away_rating, res_a, exp_a)

            self.teams.to_csv("data/predictions/scores/teams.csv", index=False)

    def predict(self, games, hfa=50):
        """ Predict the result of games

        Args:
            games (pf.DataFrame): Fixtures
            hfa (int, optional): Home field advantage. Defaults to 50.
        """

        def hubbert(rating_a, rating_b):
            """ Adjustment to the odds to add draws

            Args:
                rating_a (int): Team rating
                rating_b (int): Team rating

            Returns:
                float: Odds of a draw
            """
            return (
                np.exp(- (rating_a - rating_b) / 10) /
                np.power(1 + np.exp(- (rating_a - rating_b) / 10), 2))

        def synthesize_odds(row):
            """ Lambda function that parses row by row to compute score matrix

            Args:
                row (array): Fixture

            Returns:
                (tuple): Home and Away ratings and winning odds
            """
            # Get match data
            try:
                home_rating = self.teams.loc[
                    self.teams.team == row['team1']]['rating'].values[0]
            except:
                home_rating = 1375
                self.teams = self.teams.append(
                    {
                        'team': row['team1'],
                        'rating': home_rating,
                        'team_index': self.league_size},
                    ignore_index=True)
                self.league_size += 1

            try:
                away_rating = self.teams.loc[
                    self.teams.team == row['team2']]['rating'].values[0]
            except:
                away_rating = 1375
                self.teams = self.teams.append(
                    {
                        'team': row['team1'],
                        'rating': away_rating,
                        'team_index': self.league_size},
                    ignore_index=True)
                self.league_size += 1

            # Infer result
            exp_h = self.odds(home_rating + hfa, away_rating)
            exp_a = self.odds(away_rating, home_rating + hfa)
            exp_d = hubbert(home_rating, away_rating)

            exp_h = exp_h - exp_h * exp_d
            exp_a = exp_a - exp_a * exp_d

            return home_rating, away_rating, exp_h, exp_d, exp_a

        (
            games["home_rating"],
            games["away_rating"],
            games["home_win_p"],
            games["draw_p"],
            games["away_win_p"]
            ) = zip(*games.apply(lambda row: synthesize_odds(row), axis=1))

        return games

    def evaluate(self, games):
        """ Evaluate the model's prediction accuracy

        Args:
            games (pd.DataFrame): Fixtured to evaluate on

        Returns:
            pd.DataFrame: df with appended metrics
        """
        aggregate_df = self.predict(games)

        aggregate_df["winner"] = match_outcome(aggregate_df)

        aggregate_df["rps"] = aggregate_df.apply(
            lambda row: ranked_probability_score([
                row["home_win_p"], row["draw_p"],
                row["away_win_p"]], row["winner"]), axis=1)

        return aggregate_df

    def fine_tune(self, gw_data):
        """ Given pretrained model, fine-tune on given GW data

        Args:
            gw_data (row)
        """
        for index, match in gw_data.iterrows():
            # Get match data
            home_team = match['team1']
            away_team = match['team2']

            home_rating = match['home_rating']
            away_rating = match['away_rating']

            exp_h = match['home_win_p']
            exp_a = match['away_win_p']

            # Update ratings
            res_h = (
                1 if match['winner'] == 0
                else 0.5 if match['winner'] == 1
                else 0)
            res_a = (
                1 if match['winner'] == 2
                else 0.5 if match['winner'] == 1
                else 0)

            self.teams.loc[
                self.teams.team == home_team, 'rating'] = self.rating_update(
                    home_rating, res_h, exp_h)
            self.teams.loc[
                self.teams.team == away_team, 'rating'] = self.rating_update(
                    away_rating, res_a, exp_a)

    def backtest(self, season):
        """ Test the model's accuracy on past/finished games by iteratively
        training and testing on parts of the data.

        Args:
            season (int): Season for the test

        Returns:
            (float): Evaluation metric
        """
        # Get GW dates
        fixtures = (
            pd.read_csv("data/fpl_official/vaastav/data/2021-22/fixtures.csv")
            .loc[:, ['event', 'kickoff_time']])
        fixtures["kickoff_time"] = (
            pd.to_datetime(fixtures["kickoff_time"]).dt.date)

        # Merge on date
        season_games = (
            pd.read_csv(
                f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
                usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
            .rename(columns={
                "HomeTeam": "team1",
                "AwayTeam": "team2",
                "FTHG": "score1",
                "FTAG": "score2",
                "Date": "date",
                })
            .dropna())
        season_games["date"] = pd.to_datetime(
            season_games["date"], dayfirst=True).dt.date
        season_games = (
            pd.merge(
                season_games,
                fixtures,
                left_on='date',
                right_on='kickoff_time')
            .drop_duplicates()
            )

        rps_list = []

        for next_gw in range(1, get_next_gw()):
            # Run inference on the specific GW
            predictions = model.evaluate(
                season_games[season_games['event'] == next_gw])
            rps_list.append(predictions['rps'].values)

            # Update the model with the current GW
            model.fine_tune(predictions)

        return np.mean(rps_list)


if __name__ == "__main__":
    df = (
        pd.read_csv(
            f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
            usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        for season in [
            '9394', '9495', '9596', '9697', '9798',
            '9899', '9900', '0001', '0102', '0203',
            '0304', '0405', '0506', '0607', '0708',
            '0809', '0910', '1011', '1112', '1213',
            '1314', '1415', '1516', '1617', '1718',
            '1819', '1920', '2021'])

    df = (
        pd.concat(df)
        .rename(columns={
            "HomeTeam": "team1",
            "AwayTeam": "team2",
            "FTHG": "score1",
            "FTAG": "score2",
            "Date": "date",
            })
        .dropna())

    # Train model on all games up to the previous GW
    model = Elo(df)
    model.fit()

    season = '2122'
    next_gw = get_next_gw()-1

    # Get GW dates
    fixtures = (
        pd.read_csv("data/fpl_official/vaastav/data/2021-22/fixtures.csv")
        .loc[:, ['event', 'kickoff_time']])
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"]).dt.date

    # Merge on date
    season_games = (
        pd.read_csv(
            f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
            usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        .rename(columns={
            "HomeTeam": "team1",
            "AwayTeam": "team2",
            "FTHG": "score1",
            "FTAG": "score2",
            "Date": "date",
            })
        .dropna())
    season_games["date"] = (
        pd.to_datetime(season_games["date"], dayfirst=True).dt.date)
    season_games = (
        pd.merge(
            season_games,
            fixtures,
            left_on='date',
            right_on='kickoff_time')
        .drop_duplicates()
        )

    # Run inference on the specific GW
    predictions = model.predict(season_games[season_games['event'] == next_gw])
    if os.path.isfile("data/predictions/scores/elo.csv"):
        past_predictions = pd.read_csv("data/predictions/scores/elo.csv")
        (
            pd.concat(
                [
                    past_predictions,
                    predictions
                    .loc[:, [
                        'date', 'team1', 'team2', 'event', 'home_rating',
                        'away_rating', 'home_win_p', 'draw_p', 'away_win_p']]
                ],
                ignore_index=True
            ).to_csv("data/predictions/scores/elo.csv", index=False)
        )
    else:
        (
            predictions
            .loc[:, [
                'date', 'team1', 'team2', 'event', 'home_rating',
                'away_rating', 'home_win_p', 'draw_p', 'away_win_p']]
            .to_csv("data/predictions/scores/elo.csv", index=False)
        )
