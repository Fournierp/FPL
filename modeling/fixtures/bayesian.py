import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from utils import odds, clean_sheet, time_decay, score_mtx, get_next_gw
from ranked_probability_score import ranked_probability_score, match_outcome

import pymc3 as pm
import theano.tensor as tt


class Bayesian:
    """ Model scored goals at home and away as Bayesian Random variables """

    def __init__(self, games, performance='score', decay=False):
        """
        Args:
            games (pd.DataFrame): Finished games to used for training.
            performance (string): Observed performance metric to use in model
            decay (boolean): Apply time decay
        """
        teams = np.sort(np.unique(games["team1"]))
        league_size = len(teams)

        self.teams = (
            games.loc[:, ["team1"]]
            .drop_duplicates()
            .sort_values("team1")
            .reset_index(drop=True)
            .assign(team_index=np.arange(league_size))
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
            .sort_values("date")
        )

        df["date"] = pd.to_datetime(df["date"])
        df["days_since"] = (df["date"].max() - df["date"]).dt.days
        df["weight"] = time_decay(0.0001, df["days_since"]) if decay else 1

        # Handle different data to infer
        assert performance == 'score' or performance == 'xg'
        self.performance = performance

        self.games = df.loc[:, [
            f"{performance}1", f"{performance}2", "team1", "team2",
            "hg", "ag", "weight"]]
        self.games = self.games.dropna()

        if performance == 'xg':
            self.games = (
                self.games
                .rename(columns={"xg1": "score1", "xg2": "score2"})
            )

        self.goals_home_obs = self.games["score1"].values
        self.goals_away_obs = self.games["score2"].values
        self.home_team = self.games["hg"].values
        self.away_team = self.games["ag"].values
        self.w = self.games["weight"].values

        self.model = self._build_model()

    def _build_model(self):
        """ Build the model

        Returns:
            pymc3.Model: untrained model
        """
        with pm.Model() as model:
            # home advantage
            # Flat only:
            # Normal only:
            # Flat + Intercept:
            # Normal + Intercept:
            # home = pm.Flat("home")
            home = pm.Normal('home', mu=0, tau=.0001)
            intercept = pm.Normal('intercept', mu=0, tau=.0001)

            # attack ratings
            tau_att = pm.Gamma("tau_att", 0.1, 0.1)
            atts_star = pm.Normal(
                "atts_star",
                mu=0,
                tau=tau_att,
                shape=self.league_size)

            # defence ratings
            tau_def = pm.Gamma("tau_def", 0.1, 0.1)
            def_star = pm.Normal(
                "def_star",
                mu=0,
                tau=tau_def,
                shape=self.league_size)

            # apply sum zero constraints
            atts = pm.Deterministic("atts", atts_star - tt.mean(atts_star))
            defs = pm.Deterministic("defs", def_star - tt.mean(def_star))

            # calulate theta
            home_theta = tt.exp(
                intercept + home + atts[self.home_team] + defs[self.away_team])
            away_theta = tt.exp(
                intercept + atts[self.away_team] + defs[self.home_team])

            # goal expectation
            pm.Potential(
                'home_goals',
                self.w * pm.Poisson.dist(mu=home_theta).logp(
                    self.goals_home_obs)
            )
            pm.Potential(
                'away_goals',
                self.w * pm.Poisson.dist(mu=away_theta).logp(
                    self.goals_away_obs)
            )

        return model

    def fit(self):
        """Fit the model parameters"""
        with self.model:
            self.trace = pm.sample(
                2000,
                tune=1000,
                cores=6,
                return_inferencedata=False)

    def predict(self, games):
        """Predict the outcome of games

        Args:
            games (pd.DataFrame): Fixtures

        Returns:
            pd.DataFrame: Fixtures with game odds
        """
        parameter_df = (
            pd.DataFrame()
            .assign(attack=[
                np.mean([x[team] for x in self.trace["atts"]])
                for team in range(self.league_size)])
            .assign(defence=[
                np.mean([x[team] for x in self.trace["defs"]])
                for team in range(self.league_size)])
            .assign(team=np.array(self.teams.team_index.values))
        )

        fixtures_df = (
            pd.merge(games, parameter_df, left_on='hg', right_on='team')
            .rename(columns={"attack": "attack1", "defence": "defence1"})
            .merge(parameter_df, left_on='ag', right_on='team')
            .rename(columns={"attack": "attack2", "defence": "defence2"})
            .drop("team_y", axis=1)
            .drop("team_x", axis=1)
            .assign(home_adv=np.mean(self.trace["home"]))
            .assign(intercept=np.mean([x for x in self.trace["intercept"]]))
        )

        fixtures_df["score1_infered"] = np.exp(
            fixtures_df['intercept'] +
            fixtures_df["home_adv"] +
            fixtures_df["attack1"] +
            fixtures_df["defence2"])
        fixtures_df["score2_infered"] = np.exp(
            fixtures_df['intercept'] +
            fixtures_df["attack2"] +
            fixtures_df["defence1"])

        def synthesize_odds(row):
            """ Lambda function that parses row by row to compute score matrix

            Args:
                row (array): Fixture

            Returns:
                (tuple): Home and Away winning and clean sheets odds
            """
            m = score_mtx(row["score1_infered"], row["score2_infered"])

            home_win_p, draw_p, away_win_p = odds(m)
            home_cs_p, away_cs_p = clean_sheet(m)

            return home_win_p, draw_p, away_win_p, home_cs_p, away_cs_p

        (
            fixtures_df["home_win_p"],
            fixtures_df["draw_p"],
            fixtures_df["away_win_p"],
            fixtures_df["home_cs_p"],
            fixtures_df["away_cs_p"]
            ) = zip(*fixtures_df.apply(
                lambda row: synthesize_odds(row), axis=1))

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

    def backtest(self, train_games, test_season, path='', save=False):
        """ Test the model's accuracy on past/finished games by iteratively
        training and testing on parts of the data.

        Args:
            train_games (pd.DataFrame): All the training samples
            test_season (int): Season to use a test set
            path (string): Path extension to adjust to ipynb use
            save (boolean): Save predictions to disk

        Returns:
            (float): Evaluation metric
        """
        # Get training data
        self.train_games = train_games

        # Initialize model
        self.__init__(
            self.train_games[self.train_games['season'] != test_season])

        # Initial train
        self.fit()

        # Get test data
        # Separate testing based on per GW intervals
        fixtures = (
            pd.read_csv(
                f"{path}data/fpl_official/vaastav/data/2021-22/fixtures.csv")
            .loc[:, ['event', 'kickoff_time']])
        fixtures["kickoff_time"] = (
            pd.to_datetime(fixtures["kickoff_time"]).dt.date)
        # Get only EPL games from the test season
        self.test_games = (
            self.train_games
            .loc[self.train_games['league_id'] == 2411]
            .loc[self.train_games['season'] == test_season]
            .dropna()
            )
        self.test_games["kickoff_time"] = (
            pd.to_datetime(self.test_games["date"]).dt.date)
        # Merge on date
        self.test_games = pd.merge(
            self.test_games,
            fixtures,
            left_on='kickoff_time',
            right_on='kickoff_time')
        # Add the home team and away team index for running inference
        self.test_games = (
            pd.merge(
                self.test_games,
                self.teams,
                left_on="team1",
                right_on="team")
            .rename(columns={"team_index": "hg"})
            .drop(["team"], axis=1)
            .drop_duplicates()
            .merge(self.teams, left_on="team2", right_on="team")
            .rename(columns={"team_index": "ag"})
            .drop(["team"], axis=1)
            .sort_values("date")
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

                # Retrain model with the new GW added to the train set.
                self.__init__(
                    pd.concat([
                        self.train_games[
                            self.train_games['season'] != test_season],
                        self.test_games[self.test_games['event'] <= gw]
                        ])
                    .drop(columns=['ag', 'hg'])
                    )
                self.fit()

        if save:
            (
                predictions
                .loc[:, [
                    'date', 'team1', 'team2', 'event', 'hg', 'ag',
                    'attack1', 'defence1', 'attack2', 'defence2',
                    'home_adv', 'intercept',
                    'score1_infered', 'score2_infered',
                    'home_win_p', 'draw_p', 'away_win_p', 'home_cs_p',
                    'away_cs_p']]
                .to_csv(
                    f"{path}data/predictions/fixtures/bayesian" +
                    f"{'_decay' if self.decay else ''}" +
                    f"{'_xg' if self.performance == 'xg' else ''}.csv",
                    index=False)
            )

        return predictions


if __name__ == "__main__":
    with open('info.json') as f:
        season = json.load(f)['season']

    next_gw = get_next_gw()

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = df.loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]

    # Get GW dates
    fixtures = (
        pd.read_csv("data/fpl_official/vaastav/data/2021-22/fixtures.csv")
        .loc[:, ['event', 'kickoff_time']])
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"]).dt.date

    # Get only EPL games from the current season
    season_games = (
        df
        .loc[df['league_id'] == 2411]
        .loc[df['season'] == season]
        )
    season_games["kickoff_time"] = pd.to_datetime(season_games["date"]).dt.date

    # Merge on date
    season_games = (
        pd.merge(
            season_games,
            fixtures,
            left_on='kickoff_time',
            right_on='kickoff_time')
        .drop_duplicates()
        )

    # Train model on all games up to the previous GW
    model = Bayesian(
        pd.concat([
            df.loc[df['season'] != season],
            season_games[season_games['event'] < next_gw]
            ]))
    model.fit()

    # Add the home team and away team index for running inference
    season_games = (
        pd.merge(season_games, model.teams, left_on="team1", right_on="team")
        .rename(columns={"team_index": "hg"})
        .drop(["team"], axis=1)
        .drop_duplicates()
        .merge(model.teams, left_on="team2", right_on="team")
        .rename(columns={"team_index": "ag"})
        .drop(["team"], axis=1)
        .sort_values("date")
    )

    # Run inference on the specific GW
    predictions = model.evaluate(
        season_games[season_games['event'] == next_gw])
    print(predictions.rps.mean())
