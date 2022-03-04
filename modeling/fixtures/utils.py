import numpy as np
import requests

from scipy.stats import poisson


def score_mtx(home_goals, away_goals, max_goals=8):
    """ Generate the probability of every score margin

    Args:
        home_goals (float): Expected number of goals for the home team
        away_goals (float): Expected number of goals for the away team
        max_goals (int): Maximum number of goals scored. Defaults to 8.

    Returns:
        (array): Probability of scored goals
    """
    home_goals_pmf = poisson(home_goals).pmf(np.arange(0, max_goals))
    away_goals_pmf = poisson(away_goals).pmf(np.arange(0, max_goals))

    # Aggregate probabilities
    return np.outer(home_goals_pmf, away_goals_pmf)


def odds(m):
    """ Compute 1X2 odds for a game

    Args:
        m (array): Score matrix

    Returns:
        (tuple): Odds of home win, draw and away win
    """
    home = np.sum(np.tril(m, -1))
    draw = np.sum(np.diag(m))
    away = np.sum(np.triu(m, 1))
    return (home, draw, away)


def clean_sheet(m):
    """ Compute clean sheet odds of home and away

    Args:
        m (array): Score matrix

    Returns:
        (tuple): Odds of home clean sheet, away clean sheet
    """
    home = np.sum(m[:, 0])
    away = np.sum(m[0, :])
    return (home, away)


def get_next_gw():
    """ Scrapes the FPL api to get the next GW number

    Returns:
        (int): GW
    """
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url).json()

    # Get current gameweek
    for idx, gw in enumerate(res['events']):
        if gw['is_next']:
            return idx + 1
