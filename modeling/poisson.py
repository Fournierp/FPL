import pandas as pd
import numpy as np

from scipy.stats import poisson
from scipy.optimize import minimize

from utils import odds, clean_sheet, score_mtx
from ranked_probability_score import ranked_probability_score, match_outcome


class Poisson:

    def __init__(self, games):
        self.games = games.loc[:, ["score1", "score2", "team1", "team2"]]
        self.games["score1"] = self.games["score1"].astype(int)
        self.games["score2"] = self.games["score2"].astype(int)

        self.teams = np.sort(np.unique(self.games["team1"]))
        self.league_size = len(self.teams)

        self.parameters = np.concatenate(
            (
                np.repeat(1, self.league_size), # Attack strength
                np.repeat(1, self.league_size), # Defense strength
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
        aggregate_df["loglikelihood"] = aggregate_df["score1_loglikelihood"] + aggregate_df["score2_loglikelihood"]
        
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
        parameter_df.to_csv("poisson_parameters.csv")


    def load_parameters(self):
        parameter_df = pd.read_csv("poisson_parameters.csv")
        
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


if __name__ == "__main__":

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        .dropna()
        )

    poisson_model = Poisson(df[df['season'] != 2021])
    poisson_model.optimize()
    print(poisson_model.predict(df[df['season'] == 2021]))