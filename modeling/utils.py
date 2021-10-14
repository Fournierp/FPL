import numpy as np

from scipy.stats import poisson


def score_mtx(home_goals, away_goals, max_goals=8):
    home_goals_pmf = poisson(home_goals).pmf(np.arange(0, max_goals))
    away_goals_pmf = poisson(away_goals).pmf(np.arange(0, max_goals))

    # Aggregate probabilities
    return np.outer(home_goals_pmf, away_goals_pmf)


def odds(m):
    home = np.sum(np.tril(m, -1))
    draw = np.sum(np.diag(m))
    away = np.sum(np.triu(m, 1))
    return (home, draw, away)


def clean_sheet(m):
    home = np.sum(m[:, 0])
    away = np.sum(m[0, :])
    return (home, away)


def time_decay(xi, t):
    return np.exp(-xi * t)