import pandas as pd
import numpy as np
import json

from bayesian import Bayesian
from utils import score_mtx, odds, clean_sheet

import matplotlib.pyplot as plt
import matplotlib as mpl
from highlight_text import fig_text


def home_points(df):
    """ Compute points scored at home

    Args:
        df (pd.DataFrame): Fixtures

    Returns:
        (pd.Series): Column with the points scored at home
    """
    return np.select(
        [
            df["score1"] > df["score2"],
            df["score1"] == df["score2"],
            df["score1"] < df["score2"],
        ],
        [
            3,
            1,
            0,
        ],
        default=1,
    )


def away_points(df):
    """ Compute points scored away

    Args:
        df (pd.DataFrame): Fixtures

    Returns:
        (pd.Series): Column with the points scored at away
    """
    return np.select(
        [
            df["score1"] > df["score2"],
            df["score1"] == df["score2"],
            df["score1"] < df["score2"],
        ],
        [
            0,
            1,
            3,
        ],
        default=1,
    )


def current_table():
    """ Copmute the current league table

    Returns:
        (pd.DataFrame): Current league table
    """
    fixtures = pd.read_csv(
        '../data/fpl_official/vaastav/data/2021-22/fixtures.csv')
    results = (
        fixtures[fixtures['finished']]
        .loc[:, ['team_a', 'team_a_score', 'team_h', 'team_h_score', 'event']]
        .rename(columns={
            'team_h_score': 'score1',
            'team_a_score': 'score2',
            }))

    league_table = (
        pd.read_csv('../data/fpl_official/vaastav/data/2021-22/teams.csv')
        [['id', 'name', 'short_name']])
    league_table['Points'] = 0

    games = pd.read_csv("../data/fivethirtyeight/spi_matches.csv")
    games = (
        games
        .loc[games['league_id'] == 2411]
        .loc[games['season'] == 2021]
        .dropna()
        )

    def aggregate_points(team):
        return home_results[home_results['team_h'] == team]['Points'].sum() +\
                away_results[away_results['team_a'] == team]['Points'].sum()

    home_results = results.copy()
    home_results['Points'] = home_results.apply(
        lambda row: home_points(row), axis=1)
    home_results = (home_results.loc[:, [
        'team_a', 'team_h', 'Points', 'event']])

    away_results = results.copy()
    away_results['Points'] = away_results.apply(
        lambda row: away_points(row), axis=1)
    away_results = (away_results.loc[:, [
        'team_a', 'team_h', 'Points', 'event']])

    league_table['Points'] = league_table.id.map(aggregate_points)
    league_table['Points_h'] = league_table.id.map(
        lambda team: home_results[
            home_results['team_h'] == team]['Points'].sum())
    league_table['Points_a'] = league_table.id.map(
        lambda team: away_results[
            away_results['team_a'] == team]['Points'].sum())

    league_table = league_table.replace({
        'Brighton': 'Brighton and Hove Albion',
        'Leicester': 'Leicester City',
        'Leeds': 'Leeds United',
        'Man City': 'Manchester City',
        'Man Utd': 'Manchester United',
        'Norwich': 'Norwich City',
        'Spurs': 'Tottenham Hotspur',
        'West Ham': 'West Ham United',
        'Wolves': 'Wolverhampton'
    })

    league_table = (
        pd.merge(
            league_table,
            games.groupby('team1').sum(),
            left_on='name',
            right_on='team1')
        .loc[:, ['name', 'short_name', 'Points',
                 'Points_h', 'Points_a', 'xg1', 'xg2']]
        .rename(columns={
            'xg1': 'xG_h',
            'xg2': 'xGA_h',
            })
        .merge(games.groupby('team2').sum(), left_on='name', right_on='team2')
        .rename(columns={
            'xg1': 'xGA_a',
            'xg2': 'xG_a',
            })
        .loc[:, ['name', 'short_name', 'Points',
                 'Points_h', 'Points_a', 'xG_h', 'xGA_h', 'xG_a', 'xGA_a']]
    )
    league_table['xG'] = league_table['xG_h'] + league_table['xG_a']
    league_table['xGA'] = league_table['xGA_h'] + league_table['xGA_a']

    return league_table


def deterministic_projection(season, league_table, games, model):
    """ Generate a Monte Carlo Simulation based on the average parameters of teams

    Args:
        season (int): Current season to generate
        league_table (pd.DataFrame): Current league standing
        games (pd.DataFrame): Next games to simulate
        model (pymc3.Model): Trained model

    Returns:
        (pd.DataFrame): Simulated season
    """
    games = (
        games
        .loc[games['league_id'] == 2411]
        .loc[games['season'] == season]
        )

    fixtures = games[games.isna().any(axis=1)]

    fixtures = (
        pd.merge(fixtures, model.teams, left_on="team1", right_on="team")
        .rename(columns={"team_index": "hg"})
        .drop(["team"], axis=1)
        .merge(model.teams, left_on="team2", right_on="team")
        .rename(columns={"team_index": "ag"})
        .drop(["team"], axis=1)
        .sort_values("date")
    )

    # Predict next games
    preds = model.predict(fixtures).drop(["score1", "score2"], axis=1)
    preds = preds.rename(columns={
        'score1_infered': 'score1',
        'score2_infered': 'score2',
    })
    preds['score1'] = preds['score1'].round(2)
    preds['score2'] = preds['score2'].round(2)

    # Format table
    preds['pts_h'] = home_points(preds)
    preds['pts_a'] = away_points(preds)
    preds = preds.loc[:, ['team1', 'team2', 'score1', 'score2',
                          'pts_h', 'pts_a']]
    preds.groupby('team2').sum().rename(columns={
        'score1': 'xG_Proj_h',
        'score2': 'xGA_Proj_h',
    })

    return (
        pd.merge(
            league_table,
            preds.groupby('team1').sum().rename(
                columns={
                    'score1': 'xG_Proj_h',
                    'score2': 'xGA_Proj_h',
                    }).drop(['pts_a'], axis=1),
            left_on='name',
            right_on='team1')
        .merge(preds.groupby('team2').sum().rename(
            columns={
                'score1': 'xGA_Proj_a',
                'score2': 'xG_Proj_a',
                }).drop(['pts_h'], axis=1),
               left_on='name', right_on='team2')
        )


def stochastic_projections(season, league_table, games, model, n):
    """ Generate a Monte Carlo Simulation based on the sampling of
     outcomes from the parameters distribution of teams

    Args:
        season (int): Current season to generate
        league_table (pd.DataFrame): Current league standing
        games (pd.DataFrame): Next games to simulate
        model (pymc3.Model): Trained model
        n (int): Number of simulations to run

    Returns:
        (pd.DataFrame): Simulated season
    """
    def prediction(model, games):
        # Sample parameters
        parameter_df = (
            pd.DataFrame()
            .assign(attack=model.trace['atts'][
                np.random.randint(0, model.trace['atts'].shape[0]), :])
            .assign(defence=model.trace['defs'][
                np.random.randint(0, model.trace['defs'].shape[0]), :])
            .assign(team=np.array(model.teams.team_index.values))
        )

        aggregate_df = (
            pd.merge(games, parameter_df, left_on='hg', right_on='team')
            .rename(columns={"attack": "attack1", "defence": "defence1"})
            .merge(parameter_df, left_on='ag', right_on='team')
            .rename(columns={"attack": "attack2", "defence": "defence2"})
            .drop("team_y", axis=1)
            .drop("team_x", axis=1)
            .assign(home_adv=model.trace['home'][
                np.random.randint(0, model.trace['home'].shape[0])])
            .assign(intercept=model.trace['intercept'][
                np.random.randint(0, model.trace['intercept'].shape[0])])
        )

        # Inference
        aggregate_df["score1_infered"] = np.exp(
            aggregate_df['intercept'] +
            aggregate_df["home_adv"] +
            aggregate_df["attack1"] +
            aggregate_df["defence2"])
        aggregate_df["score2_infered"] = np.exp(
            aggregate_df['intercept'] +
            aggregate_df["attack2"] +
            aggregate_df["defence1"])

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
            ) = zip(*aggregate_df.apply(
                lambda row: synthesize_odds(row), axis=1))

        return aggregate_df

    # Predict next games
    games = (
        games
        .loc[games['league_id'] == 2411]
        .loc[games['season'] == 2021]
        )

    fixtures = games[games.isna().any(axis=1)]

    fixtures = (
        pd.merge(fixtures, model.teams, left_on="team1", right_on="team")
        .rename(columns={"team_index": "hg"})
        .drop(["team"], axis=1)
        .merge(model.teams, left_on="team2", right_on="team")
        .rename(columns={"team_index": "ag"})
        .drop(["team"], axis=1)
        .sort_values("date")
    )

    dfs = []
    for i in range(n):
        table = league_table.copy()
        table['iteration'] = i

        preds = prediction(model, fixtures).drop(["score1", "score2"], axis=1)
        preds = preds.rename(columns={
            'score1_infered': 'score1',
            'score2_infered': 'score2',
        })
        preds['score1'] = preds['score1'].round(2)
        preds['score2'] = preds['score2'].round(2)

        # Format table
        preds['pts_h'] = home_points(preds)
        preds['pts_a'] = away_points(preds)
        preds = preds.loc[:, [
            'team1', 'team2', 'score1', 'score2', 'pts_h', 'pts_a']]
        preds.groupby('team2').sum().rename(columns={
            'score1': 'xG_Proj_h',
            'score2': 'xGA_Proj_h',
        })

        table = (
            pd.merge(
                table,
                preds.groupby('team1').sum().rename(
                    columns={
                        'score1': 'xG_Proj_h',
                        'score2': 'xGA_Proj_h',
                    }).drop(['pts_a'], axis=1),
                left_on='name', right_on='team1')
            .merge(preds.groupby('team2').sum().rename(
                columns={
                    'score1': 'xGA_Proj_a',
                    'score2': 'xG_Proj_a',
                    }).drop(['pts_h'], axis=1),
                   left_on='name', right_on='team2')
        )

        table['Points'] = table['pts_h'] + table['pts_a'] + table['Points']
        table['xG'] = table['xG'] + table['xG_Proj_h'] + table['xG_Proj_a']
        table['xGA'] = table['xGA'] + table['xGA_Proj_h'] + table['xGA_Proj_a']
        dfs.append(
            table
            .loc[:, ['name', 'short_name', 'Points', 'xG', 'xGA', 'iteration']]
            .sort_values(by=['Points'], ascending=False)
            .assign(position=np.arange(1, 21)))

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":

    with open('info.json') as f:
        season = json.load(f)['season']

    # Get the current league table
    league_table = current_table()

    # Fit model
    games = pd.read_csv("data/fivethirtyeight/spi_matches.csv")
    games = (
        games
        .loc[(games['league_id'] == 2411) | (games['league_id'] == 2412)]
        )

    model = Bayesian(games.dropna())
    model.fit()

    # print(deterministic_projection(season, league_table, games, model))

    n = 250
    lt = stochastic_projections(season, league_table, games, model, n)

    def percent_finish(row, pos):
        """ Computes the percentage of times a team finished at a
        given position in the simulations

        Args:
            row (array):
            pos (int): Position it might have finished at

        Returns:
            (float): percentage
        """
        return (
            np.count_nonzero(
                lt.position[lt.name == row.name].values == pos) / n)

    heatmap = pd.DataFrame(
        columns=np.arange(1, 21),
        index=lt.name.unique())

    for pos in range(1, 21):
        heatmap.loc[:, pos] = heatmap.apply(
            lambda row: percent_finish(row, pos), axis=1)

    mpl.rcParams['figure.dpi'] = 400

    body_font = "Open Sans"
    watermark_font = "DejaVu Sans"
    text_color = "w"
    background = "#282B2F"
    title_font = "DejaVu Sans"

    mpl.rcParams['xtick.color'] = text_color
    mpl.rcParams['ytick.color'] = text_color
    mpl.rcParams['text.color'] = text_color
    mpl.rcParams['axes.edgecolor'] = text_color
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_facecolor(background)
    ax.patch.set_alpha(0)

    column_headers = heatmap.columns
    row_headers = heatmap.index

    background_rgb = [40/255, 43/255, 47/255, 1]
    ccolors = np.repeat([background_rgb], len(column_headers), 0)
    rcolors = np.repeat([background_rgb], len(row_headers), 0)

    cell_text = []
    for row in heatmap.values:
        cell_text.append([f'{x*100:1.2f}' if x*100 > 0.1 else '' for x in row])

    the_table = ax.table(
        cellText=cell_text,
        rowLabels=row_headers,
        rowColours=rcolors,
        rowLoc='right',
        colColours=ccolors,
        colLabels=column_headers,
        loc='center right',
        colWidths=[.04]*20,
        fontsize=12
        )
    the_table.scale(1, 1.2)

    for i in range(1, 21):
        for j in range(0, 20):
            the_table[(i, j)].set_facecolor(background)
            the_table[(i, j)].set_edgecolor('w')
            the_table[(i, j)].set_linewidth(0.5)

            the_table[(i, j)].set_facecolor('#B82A2A')
            the_table[(i, j)].set_alpha(
                heatmap.values[i-1][j])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    fig_text(
        x=0.2, y=0.925,
        s="Premier League 2021-22: Projected League Table",
        fontweight="regular", fontsize=12, fontfamily=title_font,
        color=text_color, alpha=1)

    fig.text(
        0.8, 0.1, "Created by Paul Fournier",
        fontstyle="italic", fontsize=6, fontfamily=watermark_font,
        color=text_color)

    plt.savefig('ss.png')
