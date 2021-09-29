import pandas as pd
import numpy as np

from scipy.stats import poisson
from scipy.optimize import minimize

from datetime import datetime

from utils import odds, clean_sheet, time_decay


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
                np.repeat(-1, len(self.teams)), # Defense strength
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


    def score_mtx(self, team1, team2, max_goals=8):
        # Get the corresponding model parameters
        home_idx = np.where(self.teams == team1)[0][0]
        away_idx = np.where(self.teams == team2)[0][0]

        home = self.parameters[[home_idx, home_idx + self.league_size]]
        away = self.parameters[[away_idx, away_idx + self.league_size]]
        home_attack, home_defence = home[0], home[1]
        away_attack, away_defence = away[0], away[1]

        home_advantage = self.parameters[-1]

        # PMF
        home_goals = np.exp(home_advantage + home_attack - away_defence)
        away_goals = np.exp(away_attack - home_defence)
        home_goals_pmf = poisson(home_goals).pmf(np.arange(0, max_goals))
        away_goals_pmf = poisson(away_goals).pmf(np.arange(0, max_goals))

        # Aggregate probabilities
        return np.outer(home_goals_pmf, away_goals_pmf)


if __name__ == "__main__":

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        .dropna()
        )
    df = df[df['season'] != 2021]

    poisson_model = Poisson_Time_Decay(games)
    poisson_model.optimize()
    mtx = poisson_model.score_mtx("Arsenal", "Burnley", 6)
    print(mtx)
    print(odds(mtx))
    print(clean_sheet(mtx))