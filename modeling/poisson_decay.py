import pandas as pd
import numpy as np
import json
import os

from scipy.stats import poisson
from scipy.optimize import minimize

from datetime import datetime

from utils import odds, clean_sheet, time_decay, score_mtx, get_next_gw
from ranked_probability_score import ranked_probability_score, match_outcome
import warnings
warnings.filterwarnings('ignore')

from git import Git

class Poisson_Time_Decay:

    def __init__(self, games):
        self.games = games
        self.games["date"] = pd.to_datetime(self.games["date"])
        self.games["days_since"] = (self.games["date"].max() - self.games["date"]).dt.days
        self.games["weight"] = time_decay(0.001, self.games["days_since"])
        self.games = self.games.loc[:, ["score1", "score2", "team1", "team2", "weight"]]
        self.games["score1"] = self.games["score1"].astype(int)
        self.games["score2"] = self.games["score2"].astype(int)

        self.teams = np.sort(np.unique(self.games["team1"]))
        self.league_size = len(self.teams)

        self.parameters = np.concatenate(
            (
                np.repeat(1, len(self.teams)), # Attack strength
                np.repeat(1, len(self.teams)), # Defense strength
                [.3], # Home advantage
            )
        )


    def fit(self, parameters, games):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=parameters[:self.league_size])
            .assign(defence=parameters[self.league_size : self.league_size * 2])
            .assign(team=self.teams)
        )

        aggregate_df = (
            games.merge(parameter_df, left_on='team1', right_on='team')
            .rename(columns={"attack": "attack1", "defence": "defence1"})
            .merge(parameter_df, left_on='team2', right_on='team')
            .rename(columns={"attack": "attack2", "defence": "defence2"})
            .drop("team_y", axis=1)
            .drop("team_x", axis=1)
            .assign(home_adv=parameters[-1])
        )

        aggregate_df["score1_infered"] = np.exp(aggregate_df["home_adv"] + aggregate_df["attack1"] - aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(aggregate_df["attack2"] - aggregate_df["defence1"])

        aggregate_df["score1_loglikelihood"] = poisson.logpmf(aggregate_df["score1"], aggregate_df["score1_infered"])
        aggregate_df["score2_loglikelihood"] = poisson.logpmf(aggregate_df["score2"], aggregate_df["score2_infered"])
        aggregate_df["loglikelihood"] = (aggregate_df["score1_loglikelihood"] + aggregate_df["score2_loglikelihood"]) * aggregate_df['weight']
        
        return -aggregate_df["loglikelihood"].sum()


    def optimize(self):
        # Set the home rating to have a unique set of values for reproducibility
        constraints = [{"type": "eq", "fun": lambda x: sum(x[: self.league_size]) - self.league_size}]

        # Set the maximum and minimum values the parameters of the model can take
        bounds = [(0, 3)] * self.league_size * 2
        bounds += [(0, 1)]

        self.solution = minimize(
            self.fit,
            self.parameters,
            args=self.games,
            constraints=constraints,
            bounds=bounds)

        self.parameters = self.solution["x"]


    def print_parameters(self):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=self.parameters[:self.league_size])
            .assign(defence=self.parameters[self.league_size : self.league_size * 2])
            .assign(team=self.teams)
            .assign(home_adv=self.parameters[-1])
        )
        return parameter_df


    def save_parameters(self):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=self.parameters[:self.league_size])
            .assign(defence=self.parameters[self.league_size : self.league_size * 2])
            .assign(team=self.teams)
            .assign(home_adv=self.parameters[-1])
        )
        parameter_df.to_csv("poisson_decay_parameters.csv")


    def load_parameters(self):
        parameter_df = pd.read_csv("poisson_decay_parameters.csv")
        
        self.league_size = (parameter_df.shape[0] - 1) / 2
        self.teams = parameter_df.loc[:, 'team']
        self.parameters = (
            parameter_df.loc[:, 'attack'].
            append(parameter_df.loc[:, 'defence']).
            append(parameter_df.loc[:, 'home_adv'])
            )


    def get_team_parameters(self, team):
        idx = np.where(self.teams == team)[0][0]

        parameters = self.parameters[[idx, idx + self.league_size]]
        attack, defence = parameters[0], parameters[1]

        home_advantage = self.parameters[-1]

        return attack, defence, home_advantage


    def predict(self, games):
        """ Predict score for several fixtures. """
        parameter_df = (
            pd.DataFrame()
            .assign(attack=self.parameters[:self.league_size])
            .assign(defence=self.parameters[self.league_size : self.league_size * 2])
            .assign(team=self.teams)
        )

        aggregate_df = (
            games.merge(parameter_df, left_on='team1', right_on='team')
            .rename(columns={"attack": "attack1", "defence": "defence1"})
            .merge(parameter_df, left_on='team2', right_on='team')
            .rename(columns={"attack": "attack2", "defence": "defence2"})
            .drop("team_y", axis=1)
            .drop("team_x", axis=1)
            .assign(home_adv=self.parameters[-2])
            .assign(rho=self.parameters[-1])
        )

        aggregate_df["score1_infered"] = np.exp(aggregate_df["home_adv"] + aggregate_df["attack1"] - aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(aggregate_df["attack2"] - aggregate_df["defence1"])

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
        self.optimize()

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
        idx = (
            pd.DataFrame()
            .assign(team=self.teams)
            .assign(team_index=np.arange(self.league_size)))
        self.test_games = (
            self.test_games.merge(idx, left_on="team1", right_on="team")
            .rename(columns={"team_index": "hg"})
            .drop(["team"], axis=1)
            .drop_duplicates()
            .merge(idx, left_on="team2", right_on="team")
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
                    .drop(columns=['ag', 'hg']))
                self.optimize()

        return np.mean(rps_list)



if __name__ == "__main__":

    with open('info.json') as f:
        season = json.load(f)['season']

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
    model = Poisson_Time_Decay(
        pd.concat([
            df.loc[df['season'] != season],
            season_games[season_games['event'] < next_gw]
            ]))
    model.optimize()

    # Add the home team and away team index for running inference
    idx = (
        pd.DataFrame()
        .assign(team=model.teams)
        .assign(team_index=np.arange(model.league_size)))
    season_games = (
        season_games.merge(idx, left_on="team1", right_on="team")
        .rename(columns={"team_index": "hg"})
        .drop(["team"], axis=1)
        .drop_duplicates()
        .merge(idx, left_on="team2", right_on="team")
        .rename(columns={"team_index": "ag"})
        .drop(["team"], axis=1)
        .sort_values("date")
    )

    # Run inference on the specific GW
    predictions = model.predict(season_games[season_games['event'] == next_gw])
    if os.path.isfile("data/predictions/scores/poisson_decay.csv"):
        past_predictions = pd.read_csv("data/predictions/scores/poisson_decay.csv")
        (
            pd.concat(
                [
                    past_predictions,
                    predictions
                    .loc[:, [
                        'date', 'team1', 'team2', 'event', 'hg', 'ag', 'attack1', 'defence1',
                        'attack2', 'defence2', 'home_adv', 'rho', 'score1_infered',
                        'score2_infered', 'home_win_p', 'draw_p', 'away_win_p', 'home_cs_p', 'away_cs_p']]
                ],
                ignore_index=True
            ).to_csv("data/predictions/scores/poisson_decay.csv", index=False)
        )
    else:
        (
            predictions
            .loc[:, [
                'date', 'team1', 'team2', 'event', 'hg', 'ag', 'attack1', 'defence1',
                'attack2', 'defence2', 'home_adv', 'rho', 'score1_infered',
                'score2_infered', 'home_win_p', 'draw_p', 'away_win_p', 'home_cs_p', 'away_cs_p']]
            .to_csv("data/predictions/scores/poisson_decay.csv", index=False)
        )

    if len(sys.argv) > 1:
        logger.info("Saving data ...")
        Git()
    else:
        print("Local")