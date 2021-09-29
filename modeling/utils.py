import numpy as np


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