import pandas as pd
import numpy as np

from scipy.stats import poisson
from scipy.optimize import minimize


class Poisson:

    def __init__(self, games):
        self.games = games

        self.teams = np.sort(np.unique(self.games["team1"]))

        self.parameters = np.concatenate(
            (
                np.repeat(1, len(self.teams)), # Attack strength
                np.repeat(-1, len(self.teams)), # Defense strength
                [.3], # Home advantage
            )
        )


    def score_inference(self, parameters, games, teams):
        parameter_df = (
            pd.DataFrame()
            .assign(attack=parameters[:len(self.teams)])
            .assign(defence=parameters[len(self.teams) : len(self.teams) * 2])
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

        aggregate_df["score1_infered"] = np.exp(aggregate_df["home_adv"] + aggregate_df["attack1"] + aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(aggregate_df["attack2"] + aggregate_df["defence1"])

        aggregate_df["score1_loglikelihood"] = poisson.logpmf(aggregate_df["score1"], aggregate_df["score1_infered"])
        aggregate_df["score2_loglikelihood"] = poisson.logpmf(aggregate_df["score2"], aggregate_df["score2_infered"])
        aggregate_df["loglikelihood"] = aggregate_df["score1_loglikelihood"] + aggregate_df["score2_loglikelihood"]
        
        return -aggregate_df["loglikelihood"].sum()


    def optimize(self):
        # Set the home rating to have a unique set of values for reproducibility
        constraints = [{"type": "eq", "fun": lambda x: sum(x[: len(self.teams)]) - len(self.teams)}]

        # Set the maximum and minimum values the parameters of the model can take
        bounds = [(0, 3)] * len(self.teams)
        bounds += [(-3, 0)] * len(self.teams)
        bounds += [(0, 1)]

        self.solution = minimize(
            self.score_inference,
            self.parameters,
            args=(self.games, self.teams),
            constraints=constraints,
            bounds=bounds
            )

        self.parameters = self.solution["x"]


    def score_mtx(self, home_team, away_team, max_goals=8):
        # Get the corresponding model parameters
        home_idx = np.where(teams == home_team)[0][0]
        away_idx = np.where(teams == away_team)[0][0]

        home_attack, home_defence = parameters[[home_idx, home_idx + len(teams)]]
        away_attack, away_defence = parameters[[away_idx, away_idx + len(teams)]]

        home_advantage = parameters[-1]

        # PMF
        home_goals = np.exp(home_advantage + home_attack + away_defence)
        away_goals = np.exp(away_attack + home_defence)
        home_goals_vector = poisson(home_goals).pmf(np.arange(0, max_goals))
        away_goals_vector = poisson(away_goals).pmf(np.arange(0, max_goals))

        # Aggregate probabilities
        m = np.outer(home_goals_vector, away_goals_vector)
        return m


    def odds(self, m):
        home = np.sum(np.tril(m, -1))
        draw = np.sum(np.diag(m))
        away = np.sum(np.triu(m, 1))
        return f"Home: {home:.2f}, Draw {draw:.2f}, Away {away:.2f}"


    def clean_sheet(self, m):
        home = np.sum(m[:, 0])
        away = np.sum(m[0, :])
        return f"Home: {home:.2f}, Away {away:.2f}"


if __name__ == "__main__":

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        .dropna()
        )
    df = df[df['season'] != 2021]

    games = df.loc[:, ["score1", "score2", "team1", "team2"]]
    games["score1"] = games["score1"].astype(int)
    games["score2"] = games["score2"].astype(int)

    poisson_model = Poisson(games)
    poisson_model.optimize()