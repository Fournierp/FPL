import requests
import pandas as pd
import numpy as np
import sasoptpy as so

import warnings
warnings.filterwarnings("ignore")


def get_team(team_id, gw):
    """ Get the players in a team

    Args:
        team_id (int): Team id to get the data from
        gw (int): GW in which the team is taken

    Returns:
        (tuple): List of integers, Remaining budget
    """
    res = requests.get(
        'https://fantasy.premierleague.com/api/entry/' +
        f'{team_id}/event/{gw}/picks/').json()

    # Scrape GW before FH to get the team from GW prior
    if res['active_chip'] == 'freehit':
        res = requests.get(
            'https://fantasy.premierleague.com/api/entry/' +
            f'{team_id}/event/{gw-1}/picks/').json()

    # Adjust the player id with fplreview indices
    return [i['element'] for i in res['picks']], res['entry_history']['bank']


def randomize(seed, df, start):
    """ Apply random noise to EV data

    Args:
        seed (int): Seed for the random number generator (for reproducibility)
        df (pd.DataFrame): EV data
        start (int): Next GW

    Returns:
        (pd.DataFrame): Randomized EV data
    """
    rng = np.random.default_rng(seed=seed)
    gws = np.arange(
        start,
        start+len([col for col in df.columns if '_Pts' in col])
        )

    for w in gws:
        noise = (
            df[f"{w}_Pts"] *
            (92 - df[f"{w}_xMins"]) / 134 *
            rng.standard_normal(size=len(df))
            )
        df[f"{w}_Pts"] = df[f"{w}_Pts"] + noise

    return df


def get_predictions(noise=False, premium=False):
    """ Load CSV file of EV Data

    Args:
        noise (bool, optional): Apply noise. Defaults to False.
        premium (bool, optional): Load premium data. Defaults to False.

    Returns:
        (pd.DataFrame): EV Data
    """
    if premium: 
        start = get_next_gw()
        df = pd.read_csv(
            f"data/fpl_review/2021-22/gameweek/{start}/fplreview_mp.csv")

        # One hot encoded values for the constraints
        if df.Pos.dtype == np.int:
            df["Pos"] = df["Pos"].map(
                {
                    1: 'G',
                    2: 'D',
                    3: 'M',
                    4: 'F'
                })

        df = pd.concat([df, pd.get_dummies(df.Pos)], axis=1)
        df = pd.concat([df, pd.get_dummies(df.Team)], axis=1)
        df = df.set_index('ID')
        df.BV = df.BV*10
        df.SV = df.SV*10

        return df.fillna(0)

    else:
        start = get_next_gw()
        df = pd.read_csv(
            f"data/fpl_review/2021-22/gameweek/{start}/fplreview_fp.csv")
        if df.Pos.dtype == np.int:
            df["Pos"] = df["Pos"].map(
                {
                    1: 'G',
                    2: 'D',
                    3: 'M',
                    4: 'F'
                })
        # One hot encoded values for the constraints
        df = pd.concat([df, pd.get_dummies(df.Pos)], axis=1)
        df = pd.concat([df, pd.get_dummies(df.Team)], axis=1)
        df = df.set_index('ID')

        return df.fillna(0)


def get_transfer_history(team_id, last_gw):
    """ Load team transfer strategy data

    Args:
        team_id (int): Team id to get the data from
        last_gw (int): GW in which the team is taken

    Returns:
        (list): List of transfers made
    """
    transfers = []
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gw in range(last_gw, 0, -1):
        res = requests.get(
            'https://fantasy.premierleague.com/api/entry/' +
            f'{team_id}/event/{gw}/picks/'
            ).json()
        transfer = res['entry_history']['event_transfers']
        chip = res['active_chip']

        if chip is not None and chip != '3xc' and chip != 'bboost':
            transfer = 2
        transfers.append(transfer)
        if transfer > 1:
            break

    return transfers


def get_rolling(team_id, last_gw):
    """ Load team transfer strategy data

    Args:
        team_id (int): Team id to get the data from
        last_gw (int): GW in which the team is taken

    Returns:
        (tuple): Rolling transfer value, Last GW transfer
    """
    transfers = get_transfer_history(team_id, last_gw)

    # Start from gw where last chip used or when hits were taken
    # Reset FT count
    rolling = 0
    for transfer in reversed(transfers):
        # Transfer logic
        rolling = min(max(rolling + 1 - transfer, 0), 1)

    return rolling, transfers[0]


def get_chips(team_id, last_gw):
    """ Get team chip strategy

    Args:
        team_id (int): Team id to get the data from
        last_gw (int): GW in which the team is taken

    Returns:
        (tuple): Availability for freehit, wildcard, bboost, threexc
    """
    freehit, wildcard, bboost, threexc = 0, 0, 0, 0
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gw in range(last_gw, 0, -1):
        res = requests.get(
            'https://fantasy.premierleague.com/api/entry/' +
            f'{team_id}/event/{gw}/picks/'
            ).json()
        chip = res['active_chip']

        if chip == '3xc':
            threexc = gw
        if chip == 'bboost':
            bboost = gw
        if chip == 'wildcard' and wildcard == 0:
            wildcard = gw
        if chip == 'freehit':
            freehit = gw

    # Handle the WC reset at GW 20
    if wildcard <= 20 and last_gw >= 20:
        wildcard = 0

    return freehit, wildcard, bboost, threexc


def get_next_gw():
    """ Get the value of the next GW to be played

    Returns:
        (int): GW value
    """
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url).json()

    for idx, gw in enumerate(res['events']):
        # if not gw['finished']:
        if gw['is_next']:
            return idx + 1


def get_ownership_data():
    """ Load CSV Ownership Data

    Returns:
        (pd.DataFrame): Ownership Data
    """
    gw = get_next_gw() - 1
    df = pd.read_csv(
        f"data/fpl_official/2021-22/gameweek/{gw}/player_ownership.csv"
        )[[
            'id', 'Top_100', 'Top_1K', 'Top_10K',
            'Top_50K', 'Top_100K', 'Top_250K']]
    df['id'] = df['id']
    return df.set_index('id')


def pretty_print(
        data,
        start,
        period,
        team,
        team_fh,
        starter,
        bench,
        captain,
        vicecaptain,
        buy,
        sell,
        free_transfers,
        hits,
        in_the_bank,
        objective_value,
        freehit,
        wildcard,
        bboost,
        threexc,
        nb_suboptimal=0):
    """ Print and save model solution

    Args:
        data (pd.DataFrame): EV and Ownership data
        start (int): Start GW
        period (int): Horizon
        team (so.Variable): Selected teams
        team_fh (so.Variable): Selected Freehit teams
        starter (so.Variable): Team that will bring points
        bench (so.Variable): Benched players
        captain (so.Variable): Captain player
        vicecaptain (so.Variable): Vicecaptain player
        buy (so.Variable): Bought players
        sell (so.Variable): Sold players
        free_transfers (so.Variable): Free transfer values
        hits (so.Variable): Penalized transfers
        in_the_bank (so.Variable): Remaining budget
        objective_value (so.Objective): Objective function
        freehit (so.Variable): Use Freehit chip
        wildcard (so.Variable): Use Wildcard chip
        bboost (so.Variable): Use bboost chip
        threexc (so.Variable): Use Threexc chip
        nb_suboptimal (int): Iteration when runing suboptimals.
    """

    df = pd.DataFrame(
        [],
        columns=[
            'GW', 'Name', 'Pos', 'Team', 'SV', 'xP', 'xMins',
            'Start', 'Bench', 'Cap', 'Vice', 'Ownership'])
    total_ev = 0
    chip_strat = []

    for w in np.arange(start, start+period):
        print(f"GW: {w} - FT: {int(free_transfers[w].get_value())}")
        for p in data.index.tolist():
            if not freehit[w].get_value():
                if team[p, w].get_value():
                    if not starter[p, w].get_value():
                        bo = [-1] + [
                            bench[p, w, o].get_value() for o in [0, 1, 2, 3]]
                    else:
                        bo = [0]

                    df = df.append(
                        {
                            'GW': w,
                            'Name': data.loc[p]['Name'],
                            'Pos': data.loc[p]['Pos'],
                            'Team': data.loc[p]['Team'],
                            'SV': data.loc[p]['SV'],
                            'xP': data.loc[p][str(w) + '_Pts'],
                            'xMins': data.loc[p][str(w) + '_xMins'],
                            'Start': int(starter[p, w].get_value()),
                            'Bench': int(np.argmax(bo)),
                            'Cap': int(captain[p, w].get_value()),
                            'Vice': int(vicecaptain[p, w].get_value()),
                            'Ownership': data.loc[p]["Top_100"]},
                        ignore_index=True)
            
            else:
                if team_fh[p, w].get_value():
                    if not starter[p, w].get_value():
                        bo = [-1] + [
                            bench[p, w, o].get_value() for o in [0, 1, 2, 3]]
                    else:
                        bo = [0]

                    df = df.append(
                        {
                            'GW': w,
                            'Name': data.loc[p]['Name'],
                            'Pos': data.loc[p]['Pos'],
                            'Team': data.loc[p]['Team'],
                            'SV': data.loc[p]['SV'],
                            'xP': data.loc[p][str(w) + '_Pts'],
                            'xMins': data.loc[p][str(w) + '_xMins'],
                            'Start': int(starter[p, w].get_value()),
                            'Bench': int(np.argmax(bo)),
                            'Cap': int(captain[p, w].get_value()),
                            'Vice': int(vicecaptain[p, w].get_value()),
                            'Ownership': data.loc[p]["Top_100"]},
                        ignore_index=True)
            if buy[p, w].get_value():
                print(f"Buy: {data.loc[p, 'Name']}")
            if sell[p, w].get_value():
                print(f"Sell: {data.loc[p, 'Name']}")

        chip = ""
        av = ""
        if freehit[w].get_value():
            chip = " - Chip: Freehit"
            chip_strat.append('FH')
        elif wildcard[w].get_value():
            chip = " - Chip: Wildcard"
            chip_strat.append('WC')
        elif bboost[w].get_value():
            chip = " - Chip: Bench Boost"
            val = (
                np.sum(df.loc[(df['GW'] == w), 'xP']) -
                np.sum(df.loc[(df['Start'] == 1) & (df['GW'] == w), 'xP'])
                )
            av = f" - Added value: {val}"
            chip_strat.append('BB')
        elif so.expr_sum(threexc[p, w] for p in data.index.tolist()).get_value():
            chip = " - Chip: Triple Captain"
            val = np.sum(df.loc[(df['Cap'] == 1) & (df['GW'] == w), 'xP'])
            av = f" - Added value: {val}"
            chip_strat.append('TC')
        else:
            chip_strat.append(None)

        xpts_val = (
            np.sum(df.loc[(df['Start'] == 1) & (df['GW'] == w), 'xP']) -
            hits[w].get_value() * 4 *
            (0 if wildcard[w].get_value() else 1) *
            (0 if freehit[w].get_value() else 1)
            )
        total_ev += xpts_val
        hits_val = (
            int(hits[w].get_value()) *
            (0 if wildcard[w].get_value() else 1) *
            (0 if freehit[w].get_value() else 1)
            )

        print("")
        print(
            df
            .loc[df.GW == w]
            .sort_values(
                by=['Pos'],
                key=lambda x: x.map({
                    'G': 0,
                    'D': 1,
                    'M': 2,
                    'F': 3})
                    )
            .sort_values(
                by=['Start', 'Bench'],
                ascending=[False, True])
                )

        print(
            f"xPts: {xpts_val:.2f} - Hits: {hits_val}" + chip + av +
            f" - ITB: {in_the_bank[w].get_value()/10:.1f}")
        print(" ____ ")

    df = (
        df
        .sort_values(
            by=['Pos'],
            key=lambda x: x.map({
                'G': 0,
                'D': 1,
                'M': 2,
                'F': 3})
            )
        .sort_values(
            by=['GW', 'Start', 'Bench'],
            ascending=[True, False, True]))
    df.to_csv(f'optimization/tmp/{nb_suboptimal}.csv')

    # print(df)
    print(f"EV: {total_ev:.2f}  |  Objective Val: {-objective_value:.2f}")

    return df, chip_strat, total_ev, -objective_value