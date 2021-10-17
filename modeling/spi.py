import pandas as pd
import numpy as np

from scipy.stats import poisson

from utils import odds, clean_sheet, score_mtx
from ranked_probability_score import ranked_probability_score, match_outcome


class SPI:

    def __init__(self, games):
        self.games = games.loc[:, ["proj_score1", "proj_score2", "score1", "score2", "team1", "team2", "prob1", "prob2", "probtie"]]


    def predict(self):
        self.games = self.games.rename(columns={
            "prob1": "home_win_p",
            "prob2": "away_win_p",
            'probtie': 'draw_p'})
        
        # Compute the CS Prob
        def synthesize_odds(row):
            m = score_mtx(row["proj_score1"], row["proj_score2"])

            home_cs_p, away_cs_p = clean_sheet(m)

            return home_cs_p, away_cs_p

        (
            self.games["home_cs_p"],
            self.games["away_cs_p"]
            ) = zip(*self.games.apply(lambda row : synthesize_odds(row), axis=1))

        return self.games


    def evaluate(self):
        """ Eval model """
        aggregate_df = self.predict()

        aggregate_df["winner"] = match_outcome(aggregate_df)

        aggregate_df["rps"] = aggregate_df.apply(lambda row: ranked_probability_score([row["home_win_p"], row["draw_p"], row["away_win_p"]], row["winner"]), axis=1)

        return aggregate_df


    def reverse_engineer_odds(self, row):
        home_goals_pmf = poisson(row["proj_score1"]).pmf(np.arange(0, 8))
        away_goals_pmf = poisson(row["proj_score2"]).pmf(np.arange(0, 8))

        m = np.outer(home_goals_pmf, away_goals_pmf)

        row["home_win_p"], row["draw_p"], row["away_win_p"] = odds(m)

        return row


if __name__ == "__main__":

    df = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    df = (df
        .loc[(df['league_id'] == 2411) | (df['league_id'] == 2412)]
        )
    df = df[df['season'] == 2021]
    df = df[df['score1'].notna()]

    spi = SPI(df)

    print(spi.predict())