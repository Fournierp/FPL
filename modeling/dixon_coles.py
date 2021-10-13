import pandas as pd
import numpy as np

from scipy.stats import poisson
from scipy.optimize import minimize

from datetime import datetime

from utils import odds, clean_sheet, time_decay, score_mtx
from ranked_probability_score import ranked_probability_score, match_outcome
import warnings
warnings.filterwarnings('ignore')


class Dixon_Coles:

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
                [.1], # Rho
            )
        )


    def _low_score_adj(self, df):
        return np.select(
            [
                (df["score1"] == 0) & (df["score2"] == 0),
                (df["score1"] == 0) & (df["score2"] == 1),
                (df["score1"] == 1) & (df["score2"] == 0),
                (df["score1"] == 1) & (df["score2"] == 1),
            ],
            [
                1 - (df["score1_infered"] * df["score2_infered"] * df["rho"]),
                1 + (df["score1_infered"] * df["rho"]),
                1 + (df["score2_infered"] * df["rho"]),
                1 - df["rho"],
            ],
            default=1,
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
            .assign(home_adv=parameters[-2])
            .assign(rho=parameters[-1])
        )

        aggregate_df["score1_infered"] = np.exp(aggregate_df["home_adv"] + aggregate_df["attack1"] - aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(aggregate_df["attack2"] - aggregate_df["defence1"])

        aggregate_df["score1_loglikelihood"] = poisson.logpmf(aggregate_df["score1"], aggregate_df["score1_infered"])
        aggregate_df["score2_loglikelihood"] = poisson.logpmf(aggregate_df["score2"], aggregate_df["score2_infered"])

        aggregate_df["adj"] = self._low_score_adj(aggregate_df)

        aggregate_df["loglikelihood"] = (
            aggregate_df["score1_loglikelihood"]
            + aggregate_df["score2_loglikelihood"]
            + np.log(aggregate_df["adj"])
            ) * aggregate_df['weight']

        return -aggregate_df["loglikelihood"].sum()


    def optimize(self):
        # Set the home rating to have a unique set of values for reproducibility
        constraints = [{"type": "eq", "fun": lambda x: sum(x[: len(self.teams)]) - len(self.teams)}]

        # Set the maximum and minimum values the parameters of the model can take
        bounds = [(0, 3)] * self.league_size * 2
        bounds += [(0, 1)]
        bounds += [(-1, 1)]

        self.solution = minimize(
            self.fit,
            self.parameters,
            args=self.games,
            constraints=constraints,
            bounds=bounds)

        self.parameters = self.solution["x"]


    def score_mtx(self, team1, team2, max_goals=8):
        """ Predict score for a single fixture. """
        # Get the corresponding model parameters
        home_idx = np.where(self.teams == team1)[0][0]
        away_idx = np.where(self.teams == team2)[0][0]

        home = self.parameters[[home_idx, home_idx + self.league_size]]
        away = self.parameters[[away_idx, away_idx + self.league_size]]
        home_attack, home_defence = home[0], home[1]
        away_attack, away_defence = away[0], away[1]

        home_advantage = self.parameters[-2]
        rho = self.parameters[-1]

        # PMF
        home_goals = np.exp(home_advantage + home_attack - away_defence)
        away_goals = np.exp(away_attack - home_defence)
        home_goals_pmf = poisson(home_goals).pmf(np.arange(0, max_goals))
        away_goals_pmf = poisson(away_goals).pmf(np.arange(0, max_goals))

        # Aggregate probabilities
        m = np.outer(home_goals_pmf, away_goals_pmf)

        # Apply Dixon and Coles adjustment
        m[0, 0] *= 1 - home_goals * away_goals * rho
        m[0, 1] *= 1 + home_goals * rho
        m[1, 0] *= 1 + away_goals * rho
        m[1, 1] *= 1 - rho
        return m


    def save_parameters(self):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=self.parameters[:self.league_size])
            .assign(defence=self.parameters[self.league_size : self.league_size * 2])
            .assign(team=self.teams)
            .assign(home_adv=self.parameters[-2])
            .assign(rho=self.parameters[-1])
        )
        parameter_df.to_csv("dixon_coles_parameters.csv")


    def load_parameters(self):
        parameter_df = pd.read_csv("dixon_coles_parameters.csv")
        
        self.league_size = (parameter_df.shape[0] - 1) / 2
        self.teams = parameter_df.loc[:, 'team']
        self.parameters = (
            parameter_df.loc[:, 'attack'].
            append(parameter_df.loc[:, 'defence']).
            append(parameter_df.loc[0, 'home_adv']).
            append(parameter_df.loc[0, 'rho'])
            )


    def print_parameters(self):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=self.parameters[:self.league_size])
            .assign(defence=self.parameters[self.league_size : self.league_size * 2])
            .assign(team=self.teams)
            .assign(home_adv=self.parameters[-2])
            .assign(rho=self.parameters[-1])
        )
        return parameter_df


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
        aggregate_df["rho"] = self.parameters[-1]

        aggregate_df["score1_infered"] = np.exp(aggregate_df["home_adv"] + aggregate_df["attack1"] - aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(aggregate_df["attack2"] - aggregate_df["defence1"])

        def synthesize_odds(row):
            m = score_mtx(row["score1_infered"], row["score2_infered"])

            m[0, 0] *= 1 - row["score1"] * row["score2"] * row["rho"]
            m[0, 1] *= 1 + row["score1"] * row["rho"]
            m[1, 0] *= 1 + row["score2"] * row["rho"]
            m[1, 1] *= 1 - row["rho"]

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


if __name__ == "__main__":

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        .dropna()
        )

    dc_model = Dixon_Coles(df[df['season'] != 2021])
    dc_model.optimize()
    print(poisson_model.predict(df[df['season'] == 2021]))