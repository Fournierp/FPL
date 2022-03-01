import numpy as np
import requests

from scipy.stats import poisson


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
