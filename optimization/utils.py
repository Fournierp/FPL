import requests
import pandas as pd
import numpy as np
import sasoptpy as so


def get_team(team_id, gw):
    res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/').json()
    # Adjust the player id with fplreview indices
    return [i['element'] for i in res['picks']], res['entry_history']['bank']


def randomize(seed, df, start):
    rng = np.random.default_rng(seed=seed)
    gws = np.arange(start, start+len([col for col in df.columns if '_Pts' in col]))

    for w in gws:
        noise = df[f"{w}_Pts"] * (92 - df[f"{w}_xMins"]) / 134 * rng.standard_normal(size=len(df))
        df[f"{w}_Pts"] = df[f"{w}_Pts"] + noise

    return df


def get_predictions(noise=False):
    start = get_next_gw()
    df = pd.read_csv(f"data/fpl_review/2021-22/gameweek/{start}/fplreview_fp.csv")
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

    return df.fillna(0)


def get_transfer_history(team_id, last_gw):
    transfers = []
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gw in range(last_gw, 0, -1):
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/').json()
        transfer = res['entry_history']['event_transfers']
        chip = res['active_chip']

        if chip is not None and chip != '3xc' and chip != 'bboost':
            transfer = 2
        transfers.append(transfer)
        if transfer > 1:
            break

    return transfers


def get_rolling(team_id, last_gw):
    transfers = get_transfer_history(team_id, last_gw)

    # Start from gw where last chip used or when hits were taken
    # Reset FT count
    rolling = 0
    for transfer in reversed(transfers):
        # Transfer logic
        rolling = min(max(rolling + 1 - transfer, 0), 1)

    return rolling, transfers[0]


def get_chips(team_id, last_gw):
    freehit, wildcard, bboost, threexc = 0, 0, 0, 0
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gw in range(last_gw, 0, -1):
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/').json()
        chip = res['active_chip']

        if chip == '3xc':
            threexc = gw
        if chip == 'bboost':
            bboost = gw
        if chip == 'wildcard' and wildcard == 0:
            wildcard = gw
        if chip == 'freehit':
            freehit = gw

    return freehit, wildcard, bboost, threexc


def get_next_gw():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url).json()
        
    for idx, gw in enumerate(res['events']):
        if not gw['finished']:
            return idx + 1


def get_ownership_data():
    gw = get_next_gw() - 1
    df = pd.read_csv(f"data/fpl_official/2021-22/gameweek/{gw}/player_ownership.csv")[['id', 'Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K']]
    df['id'] = df['id']
    return df.set_index('id')


def pretty_print(data, start, period, team, starter, bench, captain, vicecaptain, buy, sell, free_transfers, hits, freehit=-1, wildcard=-1, bboost=-1, threexc=-1, nb_suboptimal=1):
    df = pd.DataFrame([], columns=['GW', 'Name', 'Pos', 'Team', 'SV', 'xP', 'Start', 'Bench', 'Cap', 'Vice', 'Ownership'])

    for w in np.arange(start, start+period):
        print(f"GW: {w} - FT: {int(free_transfers[w].get_value())}")
        for p in data.index.tolist():
            if team[p, w].get_value():
                if not starter[p, w].get_value():
                    bo = [-1] + [bench[p, w, o].get_value() for o in [0, 1, 2, 3]]
                else:
                    bo = [0]

                df = df.append({'GW': w, 'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                                'SV': data.loc[p]['SV'], 'xP': data.loc[p][str(w) + '_Pts'], 'Start': int(starter[p, w].get_value()),
                                'Bench': int(np.argmax(bo)),
                                'Cap': int(captain[p, w].get_value()), 'Vice': int(vicecaptain[p, w].get_value()),
                                'Ownership': data.loc[p]["Top_100"]},
                               ignore_index=True)

            if buy[p, w].get_value():
                print(f"Buy: {data.loc[p, 'Name']}")
            if sell[p, w].get_value():
                print(f"Sell: {data.loc[p, 'Name']}")

        chip = ""
        av = ""
        if freehit == w-start:
            chip = " - Chip: Freehit"
        if wildcard == w-start:
            chip = " - Chip: Wildcard"
        if bboost[w].get_value():
            chip = "- Chip: Bench Boost"
            av = f" - Added value: {np.sum(df.loc[(df['GW'] == w), 'xP']) - np.sum(df.loc[(df['Start'] == 1) & (df['GW'] == w), 'xP'])}"
        if threexc == w-start:
            chip = " - Chip: Triple Captain"
            av = f" - Added value: {np.sum(df.loc[(df['Cap'] == 1) & (df['GW'] == w), 'xP'])}"

        print(f"xPts: {np.sum(df.loc[(df['Start'] == 1) & (df['GW'] == w), 'xP']) - hits[w].get_value()*4*(0 if wildcard == w-start else 1)*(0 if freehit == w-start else 1):.2f} - Hits: {int(hits[w].get_value())*(0 if wildcard == w-start else 1)*(0 if freehit == w-start else 1)}" + chip + av)
        print(" ____ ")

    custom_order = {'G': 0, 'D': 1, 'M': 2, 'F': 3}
    df.sort_values(by=['Pos'], key=lambda x: x.map(custom_order)).sort_values(by=['GW', 'Start'], ascending=[True, False]).to_csv(f'optimization/tmp/{nb_suboptimal}.csv')
    print(df.sort_values(by=['Pos'], key=lambda x: x.map(custom_order)).sort_values(by=['GW', 'Start', 'Bench'], ascending=[True, False, True]))
