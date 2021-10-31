import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from scipy.stats import poisson

from utils import odds, clean_sheet, time_decay, score_mtx, get_next_gw
from ranked_probability_score import ranked_probability_score, match_outcome

import pymc3 as pm
import theano.tensor as tt

import arviz as az
import warnings
warnings.filterwarnings('ignore')

from git import Git

class Bayesian_XG:

    def __init__(self, games, weight_value):
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
            games.merge(self.teams, left_on="team1", right_on="team")
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
        df["weight"] = time_decay(weight_value, df["days_since"])

        self.games = df.loc[:, ["xg1", "xg2", "team1", "team2", "hg", "ag", "weight"]]
        self.games = (
            self.games
            .dropna()
            .rename(columns={"xg1": "score1", "xg2": "score2"})
        )

        self.goals_home_obs = self.games["score1"].values
        self.goals_away_obs = self.games["score2"].values
        self.home_team = self.games["hg"].values
        self.away_team = self.games["ag"].values
        self.w = self.games["weight"].values

        self.model = self._build_model()


    def _build_model(self):
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
            atts_star = pm.Normal("atts_star", mu=0, tau=tau_att, shape=self.league_size)

            # defence ratings
            tau_def = pm.Gamma("tau_def", 0.1, 0.1)
            def_star = pm.Normal("def_star", mu=0, tau=tau_def, shape=self.league_size)

            # apply sum zero constraints
            atts = pm.Deterministic("atts", atts_star - tt.mean(atts_star))
            defs = pm.Deterministic("defs", def_star - tt.mean(def_star))

            # calulate theta
            home_theta = tt.exp(intercept + home + atts[self.home_team] + defs[self.away_team])
            away_theta = tt.exp(intercept + atts[self.away_team] + defs[self.home_team])

            # goal expectation
            home_points_ = pm.Potential('home_goals', self.w * pm.Poisson.dist(mu=home_theta).logp(self.goals_home_obs))
            away_points_ = pm.Potential('away_goals', self.w * pm.Poisson.dist(mu=away_theta).logp(self.goals_away_obs))
        
        return model


    def fit(self):
        with self.model:
            self.trace = pm.sample(2000, tune=1000, cores=6, return_inferencedata=False)


    def predict(self, games):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=[np.mean([x[team] for x in self.trace["atts"]]) for team in range(self.league_size)])
            .assign(defence=[np.mean([x[team] for x in self.trace["defs"]]) for team in range(self.league_size)])
            .assign(team=np.array(self.teams.team_index.values))
        )

        aggregate_df = (
            games.merge(parameter_df, left_on='hg', right_on='team')
            .rename(columns={"attack": "attack1", "defence": "defence1"})
            .merge(parameter_df, left_on='ag', right_on='team')
            .rename(columns={"attack": "attack2", "defence": "defence2"})
            .drop("team_y", axis=1)
            .drop("team_x", axis=1)
            .assign(home_adv=np.mean(self.trace["home"]))
            .assign(intercept=np.mean([x for x in self.trace["intercept"]]))
        )

        aggregate_df["score1_infered"] = np.exp(aggregate_df['intercept'] + aggregate_df["home_adv"] + aggregate_df["attack1"] + aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(aggregate_df['intercept'] + aggregate_df["attack2"] + aggregate_df["defence1"])

        def synthesize_odds(row):
            m = score_mtx(row["score1_infered"], row["score2_infered"])

            home_win_p, draw_p, away_win_p = odds(m)
            home_cs_p, away_cs_p = clean_sheet(m)

            return home_win_p, draw_p, away_win_p, home_cs_p, away_cs_p

        (
            aggregate_df["home_win_p"],
            aggregate_df["draw_p"],
            aggregate_df["away_win_p"],
            aggregate_df["home_cs_p"],
            aggregate_df["away_cs_p"]
            ) = zip(*aggregate_df.apply(lambda row : synthesize_odds(row), axis=1))

        return aggregate_df


    def evaluate(self, games):
        """ Eval model """
        aggregate_df = self.predict(games)

        aggregate_df["winner"] = match_outcome(aggregate_df)

        aggregate_df["rps"] = aggregate_df.apply(lambda row: ranked_probability_score([row["home_win_p"], row["draw_p"], row["away_win_p"]], row["winner"]), axis=1)

        return aggregate_df


    def backtest(self, train_games, test_season):

        # Get training data
        self.train_games = train_games

        # Initialize model
        self.__init__(self.train_games[self.train_games['season'] != test_season])

        # Initial train
        self.fit()

        # Get test data
        # Separate testing based on per GW intervals
        fixtures = pd.read_csv("data/fpl_official/vaastav/data/2021-22/fixtures.csv").loc[:, ['event', 'kickoff_time']]
        fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"]).dt.date
        # Get only EPL games from the test season
        self.test_games = (self.train_games
            .loc[self.train_games['league_id'] == 2411]
            .loc[self.train_games['season'] == test_season]
            .dropna()
            )
        self.test_games["kickoff_time"] = pd.to_datetime(self.test_games["date"]).dt.date
        # Merge on date
        self.test_games = self.test_games.merge(fixtures, left_on='kickoff_time', right_on='kickoff_time')
        # Add the home team and away team index for running inference
        self.test_games = (
            self.test_games.merge(self.teams, left_on="team1", right_on="team")
            .rename(columns={"team_index": "hg"})
            .drop(["team"], axis=1)
            .drop_duplicates()
            .merge(self.teams, left_on="team2", right_on="team")
            .rename(columns={"team_index": "ag"})
            .drop(["team"], axis=1)
            .sort_values("date")
        )

        rps_list = []

        for gw in range(1, 39):
            # For each GW of the season
            if gw in self.test_games['event'].values:
                
                # Run inference on the specific GW and save data.
                rps_list.append(self.evaluate(self.test_games[self.test_games['event'] == gw])['rps'].values)

                # Retrain model with the new GW added to the train set.
                self.__init__(
                    pd.concat(
                        [
                            self.train_games[self.train_games['season'] != 2021],
                            self.test_games[self.test_games['event'] <= gw]
                            ])
                    .drop(columns=['ag', 'hg'])
                    )
                self.fit()

        return np.mean(rps_list)


if __name__ == "__main__":
    with open('info.json') as f:
        season = json.load(f)['season']

    weight = 0.0001
    next_gw = get_next_gw()

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        )

    # Get GW dates
    fixtures = pd.read_csv("data/fpl_official/vaastav/data/2021-22/fixtures.csv").loc[:, ['event', 'kickoff_time']]
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"]).dt.date

    # Get only EPL games from the current season
    season_games = (df
        .loc[df['league_id'] == 2411]
        .loc[df['season'] == season]
        )
    season_games["kickoff_time"] = pd.to_datetime(season_games["date"]).dt.date

    # Merge on date
    season_games = (
        season_games
        .merge(fixtures, left_on='kickoff_time', right_on='kickoff_time')
        .drop_duplicates()
        )

    # Train model on all games up to the previous GW
    model = Bayesian_XG(
        pd.concat([
            df.loc[df['season'] != season],
            season_games[season_games['event'] < next_gw]
            ]), weight)
    model.fit()

    # Add the home team and away team index for running inference
    season_games = (
        season_games.merge(model.teams, left_on="team1", right_on="team")
        .rename(columns={"team_index": "hg"})
        .drop(["team"], axis=1)
        .drop_duplicates()
        .merge(model.teams, left_on="team2", right_on="team")
        .rename(columns={"team_index": "ag"})
        .drop(["team"], axis=1)
        .sort_values("date")
    )

    # Run inference on the specific GW
    predictions = model.predict(season_games[season_games['event'] == next_gw])
    if os.path.isfile("data/predictions/scores/bayesian_xg.csv"):
        past_predictions = pd.read_csv("data/predictions/scores/bayesian_xg.csv")
        (
            pd.concat(
                [
                    past_predictions,
                    predictions
                    .loc[:, [
                        'date', 'team1', 'team2', 'event', 'hg', 'ag', 'attack1', 'defence1',
                        'attack2', 'defence2', 'home_adv', 'intercept', 'score1_infered',
                        'score2_infered', 'home_win_p', 'draw_p', 'away_win_p', 'home_cs_p', 'away_cs_p']]
                ],
                ignore_index=True
            ).to_csv("data/predictions/scores/bayesian_xg.csv", index=False)
        )
    else:
        (
            predictions
            .loc[:, [
                'date', 'team1', 'team2', 'event', 'hg', 'ag', 'attack1', 'defence1',
                'attack2', 'defence2', 'home_adv', 'intercept', 'score1_infered',
                'score2_infered', 'home_win_p', 'draw_p', 'away_win_p', 'home_cs_p', 'away_cs_p']]
            .to_csv("data/predictions/scores/bayesian_xg.csv", index=False)
        )


    if len(sys.argv) > 1:
        logger.info("Saving data ...")
        Git()
    else:
        print("Local")