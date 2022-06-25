import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

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
                np.exp(- (rating_a - rating_b) / 100) /
                np.power(1 + np.exp(- (rating_a - rating_b) / 100), 2))

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
                # Default case when the team is promoted
                home_rating = 1375
                self.teams.loc[self.league_size, 'team'] = row['team1']
                self.teams.loc[self.league_size, 'rating'] = home_rating
                self.teams.loc[
                    self.league_size, 'team_index'] = self.league_size
                self.league_size += 1

            try:
                away_rating = self.teams.loc[
                    self.teams.team == row['team2']]['rating'].values[0]
            except:
                # Default case when the team is promoted
                away_rating = 1375
                self.teams.loc[self.league_size, 'team'] = row['team2']
                self.teams.loc[self.league_size, 'rating'] = away_rating
                self.teams.loc[
                    self.league_size, 'team_index'] = self.league_size
                self.league_size += 1

            exp_h = self.odds(home_rating + hfa, away_rating)
            exp_a = self.odds(away_rating, home_rating + hfa)
            exp_d = hubbert(home_rating, away_rating)

            exp_h = exp_h - exp_h * exp_d
            exp_a = exp_a - exp_a * exp_d
            return home_rating, away_rating, exp_h, exp_d, exp_a

        fixtures_df = games.copy()

        (
            fixtures_df["home_rating"],
            fixtures_df["away_rating"],
            fixtures_df["home_win_p"],
            fixtures_df["draw_p"],
            fixtures_df["away_win_p"]
            ) = zip(
                *fixtures_df.apply(
                    lambda row: synthesize_odds(row), axis=1
                    )
                )

        return fixtures_df

    def evaluate(self, games):
        """ Evaluate the model's prediction accuracy

        Args:
            games (pd.DataFrame): Fixtured to evaluate on

        Returns:
            pd.DataFrame: df with appended metrics
        """
        fixtures_df = self.predict(games)

        fixtures_df["winner"] = match_outcome(fixtures_df)

        fixtures_df["rps"] = fixtures_df.apply(
            lambda row: ranked_probability_score([
                row["home_win_p"], row["draw_p"],
                row["away_win_p"]], row["winner"]), axis=1)

        return fixtures_df

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

    def backtest(self, test_season, path='', save=True):
        """ Test the model's accuracy on past/finished games by iteratively
        training and testing on parts of the data.

        Args:
            test_season (string): Season to use a test set
            path (string): Path extension to adjust to ipynb use
            save (boolean): Save predictions to disk

        Returns:
            (float): Evaluation metric
        """
        df = (
            pd.read_csv(
                f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
                usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
                encoding='unicode_escape')
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
        self.__init__(df)
        self.fit()

        # Get GW dates
        fixtures = (
            pd.read_csv(
                f"{path}data/fpl_official/vaastav/data/2021-22/fixtures.csv")
            .loc[:, ['event', 'kickoff_time']])
        fixtures["kickoff_time"] = (
            pd.to_datetime(fixtures["kickoff_time"]).dt.date)

        # Merge on date
        self.test_games = (
            pd.read_csv(
                f'https://www.football-data.co.uk/mmz4281/{test_season}/E0.csv',
                usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
            .rename(columns={
                "HomeTeam": "team1",
                "AwayTeam": "team2",
                "FTHG": "score1",
                "FTAG": "score2",
                "Date": "date",
                })
            .dropna())
        self.test_games["date"] = pd.to_datetime(
            self.test_games["date"], dayfirst=True).dt.date
        self.test_games = (
            pd.merge(
                self.test_games,
                fixtures,
                left_on='date',
                right_on='kickoff_time')
            .drop_duplicates()
            )

        predictions = pd.DataFrame()

        for gw in tqdm(range(1, 39)):
            # For each GW of the season
            if gw in self.test_games['event'].values:
                # Handle case when the season is not finished

                # Run inference on the specific GW and save data.
                predictions = pd.concat([
                    predictions,
                    self.evaluate(
                        self.test_games[self.test_games['event'] == gw])
                    ])
                # Update the model with the current GW
                self.fine_tune(predictions)

        if save:
            (
                predictions
                .loc[:, [
                    'date', 'team1', 'team2', 'event', 'home_rating',
                    'away_rating', 'home_win_p', 'draw_p', 'away_win_p']]
                .to_csv(
                    f"{path}data/predictions/fixtures/elo.csv",
                    index=False)
            )

        return predictions


if __name__ == "__main__":
    if sys.argv[1] == 'predict':
        df = (
            pd.read_csv(
                f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
                usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
                encoding='unicode_escape')
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

        season = '2122'
        # Get last finished GW
        previous_gw = get_next_gw() - 2

        # Get GW dates of Last season
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

        # Train model on all games up to the previous GW
        model = Elo(
            pd.concat([
                df,
                season_games.loc[season_games['event'] <= previous_gw]
            ])
        )
        model.fit()

        # Run inference on the specific GW
        predictions = model.predict(
            season_games[season_games['event'] == previous_gw + 1])

        print(
            "Elo model's predictions for the {} games from GW{} : \n{}"\
                .format(len(predictions), previous_gw,  predictions.sort_values(by=['date'])[[
                    'date', 'event', 'team1', 'team2', 'home_win_p', 'draw_p', 'away_win_p']]))

    if sys.argv[1] == 'backtest':
        df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
        df = (
            df
            .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
            )

        model = Elo(df)
        season = '2122'
        model.backtest(season)
