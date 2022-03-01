import pandas as pd
import numpy as np
import json

from utils import get_next_gw
from ranked_probability_score import ranked_probability_score, match_outcome


class Baselines:
    """ Baselines and dummy models """

    def __init__(self, games):
        self.games = games.loc[:, ["score1", "score2", "team1", "team2"]]
        self.games = self.games.dropna()
        self.games["score1"] = self.games["score1"].astype(int)
        self.games["score2"] = self.games["score2"].astype(int)

        self.teams = np.sort(np.unique(self.games["team1"]))
        self.league_size = len(self.teams)

    def uniform(self, games):
        parameter_df = (
            pd.DataFrame()
            .assign(team=self.teams)
        )

        aggregate_df = (
            pd.merge(games, parameter_df, left_on='team1', right_on='team')
            .merge(parameter_df, left_on='team2', right_on='team')
        )

        aggregate_df["home_win_p"] = 0.333
        aggregate_df["draw_p"] = 0.333
        aggregate_df["away_win_p"] = 0.333

        return aggregate_df

    def home_bias(self, games):
        parameter_df = (
            pd.DataFrame()
            .assign(team=self.teams)
        )

        aggregate_df = (
            pd.merge(games, parameter_df, left_on='team1', right_on='team')
            .merge(parameter_df, left_on='team2', right_on='team')
        )

        aggregate_df["home_win_p"] = 1
        aggregate_df["draw_p"] = 0
        aggregate_df["away_win_p"] = 0

        return aggregate_df

    def draw_bias(self, games):
        parameter_df = (
            pd.DataFrame()
            .assign(team=self.teams)
        )

        aggregate_df = (
            pd.merge(games, parameter_df, left_on='team1', right_on='team')
            .merge(parameter_df, left_on='team2', right_on='team')
        )

        aggregate_df["home_win_p"] = 0
        aggregate_df["draw_p"] = 1
        aggregate_df["away_win_p"] = 0

        return aggregate_df

    def away_bias(self, games):
        parameter_df = (
            pd.DataFrame()
            .assign(team=self.teams)
        )

        aggregate_df = (
            pd.merge(games, parameter_df, left_on='team1', right_on='team')
            .merge(parameter_df, left_on='team2', right_on='team')
        )

        aggregate_df["home_win_p"] = 0
        aggregate_df["draw_p"] = 0
        aggregate_df["away_win_p"] = 1

        return aggregate_df

    def evaluate(self, games, function_name):
        if function_name == "uniform":
            aggregate_df = self.uniform(games)
        if function_name == "home":
            aggregate_df = self.home_bias(games)
        if function_name == "draw":
            aggregate_df = self.draw_bias(games)
        if function_name == "away":
            aggregate_df = self.away_bias(games)

        aggregate_df["winner"] = match_outcome(aggregate_df)

        aggregate_df["rps"] = aggregate_df.apply(
            lambda row: ranked_probability_score(
                [row["home_win_p"], row["draw_p"],
                 row["away_win_p"]], row["winner"]), axis=1)

        return aggregate_df


if __name__ == "__main__":
    with open('info.json') as f:
        season = json.load(f)['season']

    next_gw = get_next_gw()

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (
        df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        )

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
    baselines = Baselines(
        pd.concat([
            df.loc[df['season'] != season],
            season_games[season_games['event'] < next_gw]
            ]))

    # Add the home team and away team index for running inference
    idx = (
        pd.DataFrame()
        .assign(team=baselines.teams)
        .assign(team_index=np.arange(baselines.league_size)))
    season_games = (
        pd.merge(season_games, idx, left_on="team1", right_on="team")
        .rename(columns={"team_index": "hg"})
        .drop(["team"], axis=1)
        .drop_duplicates()
        .merge(idx, left_on="team2", right_on="team")
        .rename(columns={"team_index": "ag"})
        .drop(["team"], axis=1)
        .sort_values("date")
    )

    predictions = baselines.evaluate(season_games[season_games['event'] == next_gw], 'uniform')
    print(f"{(np.mean(predictions.rps)*100):.2f}")