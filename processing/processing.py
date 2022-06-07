import pandas as pd
import numpy as np
import os
import json


def get_raw_data(rank, path='data/fpl_official/2020-21/season/raw/'):
    f = os.path.join(path, f"managers_{rank}.json")

    chips = pd.DataFrame(
        list(pd.read_json(f, orient='index')['chips']),
        columns=['wildcard_1', 'freehit', 'bboost', 'wildcard_2', 'threexc'],
        index=pd.read_json(f, orient='index').index
        )

    # Change index type & Sort index
    chips.drop('[]', inplace=True, errors='ignore')
    chips.index = chips.index.map(int)
    chips.sort_index(inplace=True)

    chips = chips.fillna(0)
    chips = chips.astype(int)
    # Handle the cases when managers have only used their second Wildcard.
    chips.loc[(chips['wildcard_2'] == 0) & (chips['wildcard_1'] > 16), 'wildcard_2'] = chips[chips['wildcard_2'] == 0]['wildcard_1']
    chips.loc[(chips['wildcard_1'] == chips['wildcard_2']) & (chips['wildcard_1'] > 16), 'wildcard_1'] = 0

    teams = pd.DataFrame(
        list(pd.read_json(f, orient='index')['team']),
        columns=[str(gw) for gw in np.arange(1, 39)],
        index=pd.read_json(f, orient='index').index
        )
    teams.drop('[]', inplace=True, errors='ignore')

    caps = pd.DataFrame(
        list(pd.read_json(f, orient='index')['cap']),
        columns=[str(gw) for gw in np.arange(1, 39)],
        index=pd.read_json(f, orient='index').index
        )
    caps.drop('[]', inplace=True, errors='ignore')

    vice = pd.DataFrame(
        list(pd.read_json(f, orient='index')['vice']),
        columns=[str(gw) for gw in np.arange(1, 39)],
        index=pd.read_json(f, orient='index').index
        )
    vice.drop('[]', inplace=True, errors='ignore')

    bench_pts = pd.DataFrame(
        list(pd.read_json(f, orient='index')['bench_pts']),
        columns=[str(gw) for gw in np.arange(1, 39)],
        index=pd.read_json(f, orient='index').index
        )
    bench_pts.drop('[]', inplace=True, errors='ignore')

    transfers = pd.DataFrame(
        list(pd.read_json(f, orient='index')['transfers']),
        columns=[str(gw) for gw in np.arange(1, 39)],
        index=pd.read_json(f, orient='index').index
        )
    transfers.drop('[]', inplace=True, errors='ignore')

    return chips, teams, caps, vice, bench_pts, transfers


def get_season_points():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank, f'data/fpl_official/{season}-{season % 2000 + 1}/season/raw/')

        points = pd.DataFrame().reindex_like(bench_pts)

        free_transfer = np.zeros(105000)

        all_gw_data = pd.read_csv(os.path.join(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/gws/merged_gw.csv'))[['GW', 'element', 'total_points', 'minutes']]

        for gw in np.arange(1, 39):
            gw_data = all_gw_data[all_gw_data['GW'] == gw]

            for player in points.index:
                # Not registered team
                if not len(teams.loc[player, str(gw)]):
                    points.loc[player, str(gw)] = 0
                    continue

                # FPL Team
                fpl_team = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)])]

                # All player pts
                points.loc[player, str(gw)] = sum(fpl_team['total_points'])

                # Captain/Vice points
                try:
                    if chips.loc[int(player), 'threexc'] != gw:
                        multiplier = 1
                    else:
                        multiplier = 2
                    # This lookup throws an exception if the player has a BGW
                    if sum(gw_data[gw_data['element'] == caps.loc[player, str(gw)]]['minutes']) == 0:
                        # Captain does not play
                        points.loc[player, str(gw)] += multiplier * sum(fpl_team[fpl_team['element'] == vice.loc[player, str(gw)]]['total_points'].values)
                    points.loc[player, str(gw)] += multiplier * sum(fpl_team[fpl_team['element'] == caps.loc[player, str(gw)]]['total_points'].values)
                except:
                    points.loc[player, str(gw)] += multiplier * sum(fpl_team[fpl_team['element'] == vice.loc[player, str(gw)]]['total_points'].values)

                # Bench
                if chips.loc[int(player), 'bboost'] != gw:
                    points.loc[player, str(gw)] -= bench_pts.loc[player, str(gw)]

                # Hits
                try:
                    if not (chips.loc[int(player), 'freehit'] == gw or chips.loc[int(player), 'wildcard_1'] == gw or chips.loc[int(player), 'wildcard_2'] == gw) :
                        transfer = len(transfers.loc[player, str(gw)]['in'])
                        hit = free_transfer[int(player)] - transfer
                        if hit > 0:
                            # No transfer
                            free_transfer[int(player)] = 2
                        elif hit == 0 :
                            # Used all FT
                            free_transfer[int(player)] = 1
                        else:
                            # Hits
                            points.loc[player, str(gw)] += 4 * hit
                            free_transfer[int(player)] = 1

                    else:
                        free_transfer[player] = 1
                except:
                    free_transfer[int(player)] = min(2, free_transfer[int(player)] + 1)

        points = points.fillna(0)
        points = points.astype(int)
        points.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/points_{rank}.csv')


def get_season_value():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank, f'data/fpl_official/{season}-{season % 2000 + 1}/season/raw/')

        team_value = pd.DataFrame().reindex_like(bench_pts)
        in_the_bank = pd.DataFrame().reindex_like(bench_pts)
        bench_value = pd.DataFrame().reindex_like(bench_pts)

        all_gw_data = pd.read_csv(os.path.join(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/gws/merged_gw.csv'))[['GW', 'element', 'value']]

        for gw in np.arange(1, 39):
            gw_data = all_gw_data[all_gw_data['GW'] == gw]
            next_gw_data = all_gw_data[all_gw_data['GW'] == gw+1]

            for player in team_value.index:
                # Not registered team
                if not len(teams.loc[player, str(gw)]):
                    team_value.loc[player, str(gw)] = 1000
                    continue

                # FPL Team
                fpl_team = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)])].drop_duplicates(subset='element', keep="first")
                fpl_bench = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)][-4:])].drop_duplicates(subset='element', keep="first")
                next_fpl_team = next_gw_data[next_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")
                next_fpl_bench = next_gw_data[next_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)][-4:] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")

                # Team value
                # Handle missing players from DF due to BGW
                team_value.loc[player, str(gw)] = (
                    sum(fpl_team['value']) +
                    sum(next_fpl_team['value'])
                )
                bench_value.loc[player, str(gw)] = (
                    sum(fpl_bench['value']) +
                    sum(next_fpl_bench['value'])
                )
                if gw == 1 :
                    in_the_bank.loc[player, '1'] = 1000 - team_value.loc[player, '1']
                elif transfers.loc[player, str(gw)] == transfers.loc[player, str(gw)]: # NaN check
                    in_the_bank.loc[player, str(gw)] = in_the_bank.loc[player, str(gw-1)]
                    if chips.loc[int(player), 'freehit'] != gw:
                        # Huge approximative because of mid-week price changes.
                        # Get the next player value in case of BGW
                        gw_value = (
                            in_the_bank.loc[player, str(gw-1)] + 
                            sum(gw_data[gw_data['element'].isin(list(transfers.loc[player, str(gw)]['out'].values()))].drop_duplicates(subset='element', keep="first")['value']) -
                            sum(gw_data[gw_data['element'].isin(list(transfers.loc[player, str(gw)]['in'].values()))].drop_duplicates(subset='element', keep="first")['value'])
                        )

                        in_missing = [player_id for player_id in transfers.loc[player, str(gw)]['in'].values() if player_id not in gw_data['element'].values]
                        out_missing = [player_id for player_id in transfers.loc[player, str(gw)]['out'].values() if player_id not in gw_data['element'].values]

                        if len(out_missing):
                            gw_value += sum(next_gw_data[next_gw_data['element'].isin(out_missing)].drop_duplicates(subset='element', keep="first")['value'])
                        if len(in_missing) :
                            gw_value -= sum(next_gw_data[next_gw_data['element'].isin(in_missing)].drop_duplicates(subset='element', keep="first")['value'])

                        in_the_bank.loc[player, str(gw)] =  max(0, gw_value)
                else:
                    in_the_bank.loc[player, str(gw)] = in_the_bank.loc[player, str(gw-1)]

        team_value = team_value.fillna(0)
        team_value = team_value.astype(int)
        in_the_bank = in_the_bank.fillna(0)
        in_the_bank = in_the_bank.astype(int)
        bench_value = bench_value.fillna(0)
        bench_value = bench_value.astype(int)

        team_value.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/team_value_{rank}.csv')
        in_the_bank.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/in_the_bank_{rank}.csv')
        bench_value.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/bench_value_{rank}.csv')


def get_season_formation():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank, f'data/fpl_official/{season}-{season % 2000 + 1}/season/raw/')

        team_formation = pd.DataFrame().reindex_like(bench_pts)
        bench_order = pd.DataFrame().reindex_like(bench_pts)

        all_gw_data = pd.read_csv(os.path.join(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/gws/merged_gw.csv'))
        all_gw_data = all_gw_data[['position', 'element']].drop_duplicates(subset='element', keep="first")

        for gw in np.arange(1, 39):
            for player in team_formation.index:
                # Formation
                lineup = all_gw_data['element'].isin(teams.loc[player, str(gw)][:-4])
                team_formation.loc[player, str(gw)] = (
                    all_gw_data[(lineup) & (all_gw_data['position'] == 'FWD')].shape[0] + 
                    all_gw_data[(lineup) & (all_gw_data['position'] == 'MID')].shape[0] * 10 +
                    all_gw_data[(lineup) & (all_gw_data['position'] == 'DEF')].shape[0] * 100
                )
                mapping = {
                    'DEF': 1,
                    'MID': 2,
                    'FWD': 3,
                }
                try:
                    bench_order.loc[player, str(gw)] = (
                        100 * mapping[all_gw_data[all_gw_data['element'].isin([teams.loc[player, str(gw)][-3]])]['position'].values[0]] + \
                        10 * mapping[all_gw_data[all_gw_data['element'].isin([teams.loc[player, str(gw)][-2]])]['position'].values[0]] + \
                        mapping[all_gw_data[all_gw_data['element'].isin([teams.loc[player, str(gw)][-1]])]['position'].values[0]]
                    )
                except:
                    bench_order.loc[player, str(gw)] = 0

        team_formation = team_formation.fillna(0)
        team_formation = team_formation.astype(int)
        bench_order = bench_order.fillna(0)
        bench_order = bench_order.astype(int)

        team_formation.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/team_formation_{rank}.csv')
        bench_order.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/bench_order_{rank}.csv')


def get_season_pos_values():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank, f'data/fpl_official/{season}-{season % 2000 + 1}/season/raw/')

        gk_value = pd.DataFrame().reindex_like(bench_pts)
        def_value = pd.DataFrame().reindex_like(bench_pts)
        mid_value = pd.DataFrame().reindex_like(bench_pts)
        fwd_value = pd.DataFrame().reindex_like(bench_pts)

        all_gw_data = pd.read_csv(os.path.join(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/gws/merged_gw.csv'))[['GW', 'position', 'element', 'minutes', 'value']]

        for gw in np.arange(1, 39):
            prev_gw_data = all_gw_data[all_gw_data['GW'] == gw-1]
            gw_data = all_gw_data[all_gw_data['GW'] == gw]
            next_gw_data = all_gw_data[all_gw_data['GW'] == gw+1]

            for player in gk_value.index:
                # FPL Team
                prev_fpl_team = prev_gw_data[prev_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")
                fpl_team = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)])].drop_duplicates(subset='element', keep="first")
                next_fpl_team = next_gw_data[next_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")

                # Team value
                # Handle missing players from DF due to BGW
                gk_value.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'GK']['value']) +
                    sum(fpl_team[fpl_team['position'] == 'GK']['value']) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'GK']['value'])
                )
                def_value.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'DEF']['value']) +
                    sum(fpl_team[fpl_team['position'] == 'DEF']['value']) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'DEF']['value'])
                )
                mid_value.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'MID']['value']) +
                    sum(fpl_team[fpl_team['position'] == 'MID']['value']) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'MID']['value'])
                )
                fwd_value.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'FWD']['value']) +
                    sum(fpl_team[fpl_team['position'] == 'FWD']['value']) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'FWD']['value'])
                )

        gk_value = gk_value.fillna(0)
        gk_value = gk_value.astype(int)
        def_value = def_value.fillna(0)
        def_value = def_value.astype(int)
        mid_value = mid_value.fillna(0)
        mid_value = mid_value.astype(int)
        fwd_value = fwd_value.fillna(0)
        fwd_value = fwd_value.astype(int)

        gk_value.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/gk_value_{rank}.csv')
        def_value.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/def_value_{rank}.csv')
        mid_value.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/mid_value_{rank}.csv')
        fwd_value.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/fwd_value_{rank}.csv')


def get_season_assets():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank, f'data/fpl_official/{season}-{season % 2000 + 1}/season/raw/')

        gk_premiums = pd.DataFrame().reindex_like(bench_pts)
        def_premiums = pd.DataFrame().reindex_like(bench_pts)
        mid_premiums = pd.DataFrame().reindex_like(bench_pts)
        fwd_premiums = pd.DataFrame().reindex_like(bench_pts)

        gk_cheap = pd.DataFrame().reindex_like(bench_pts)
        def_cheap = pd.DataFrame().reindex_like(bench_pts)
        mid_cheap = pd.DataFrame().reindex_like(bench_pts)
        fwd_cheap = pd.DataFrame().reindex_like(bench_pts)

        all_gw_data = pd.read_csv(os.path.join(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/gws/merged_gw.csv'))[['GW', 'position', 'element', 'minutes', 'value']]

        for gw in np.arange(1, 39):
            prev_gw_data = all_gw_data[all_gw_data['GW'] == gw-1]
            gw_data = all_gw_data[all_gw_data['GW'] == gw]
            next_gw_data = all_gw_data[all_gw_data['GW'] == gw+1]

            for player in gk_premiums.index:
                # FPL Team
                prev_fpl_team = prev_gw_data[prev_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")
                fpl_team = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)])].drop_duplicates(subset='element', keep="first")
                next_fpl_team = next_gw_data[next_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")

                # Team value
                # Handle missing players from DF due to BGW
                gk_premiums.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'GK']['value'] > 55) +
                    sum(fpl_team[fpl_team['position'] == 'GK']['value'] > 55) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'GK']['value'] > 55)
                )
                def_premiums.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'DEF']['value'] >= 65) +
                    sum(fpl_team[fpl_team['position'] == 'DEF']['value'] >= 65) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'DEF']['value'] >= 65)
                )
                mid_premiums.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'MID']['value'] > 90) +
                    sum(fpl_team[fpl_team['position'] == 'MID']['value'] > 90) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'MID']['value'] > 90)
                )
                fwd_premiums.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'FWD']['value'] > 100) +
                    sum(fpl_team[fpl_team['position'] == 'FWD']['value'] > 100) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'FWD']['value'] > 100)
                )

                gk_cheap.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'GK']['value'] < 45) +
                    sum(fpl_team[fpl_team['position'] == 'GK']['value'] < 45) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'GK']['value'] < 45)
                )
                def_cheap.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'DEF']['value'] < 50) +
                    sum(fpl_team[fpl_team['position'] == 'DEF']['value'] < 50) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'DEF']['value'] < 50)
                )
                mid_cheap.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'MID']['value'] < 60) +
                    sum(fpl_team[fpl_team['position'] == 'MID']['value'] < 60) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'MID']['value'] < 60)
                )
                fwd_cheap.loc[player, str(gw)] = (
                    sum(prev_fpl_team[prev_fpl_team['position'] == 'FWD']['value'] <= 60) +
                    sum(fpl_team[fpl_team['position'] == 'FWD']['value'] <= 60) +
                    sum(next_fpl_team[next_fpl_team['position'] == 'FWD']['value'] <= 60)
                )

        gk_premiums = gk_premiums.fillna(0)
        gk_premiums = gk_premiums.astype(int)
        def_premiums = def_premiums.fillna(0)
        def_premiums = def_premiums.astype(int)
        mid_premiums = mid_premiums.fillna(0)
        mid_premiums = mid_premiums.astype(int)
        fwd_premiums = fwd_premiums.fillna(0)
        fwd_premiums = fwd_premiums.astype(int)

        gk_cheap = gk_cheap.fillna(0)
        gk_cheap = gk_cheap.astype(int)
        def_cheap = def_cheap.fillna(0)
        def_cheap = def_cheap.astype(int)
        mid_cheap = mid_cheap.fillna(0)
        mid_cheap = mid_cheap.astype(int)
        fwd_cheap = fwd_cheap.fillna(0)
        fwd_cheap = fwd_cheap.astype(int)

        gk_premiums.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/gk_premiums_{rank}.csv')
        def_premiums.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/def_premiums_{rank}.csv')
        mid_premiums.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/mid_premiums_{rank}.csv')
        fwd_premiums.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/fwd_premiums_{rank}.csv')

        gk_cheap.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/gk_cheap_{rank}.csv')
        def_cheap.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/def_cheap_{rank}.csv')
        mid_cheap.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/mid_cheap_{rank}.csv')
        fwd_cheap.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/fwd_cheap_{rank}.csv')


def get_season_transfers():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank, f'data/fpl_official/{season}-{season % 2000 + 1}/season/raw/')

        hit_points = pd.DataFrame().reindex_like(bench_pts)
        free_transfers = pd.DataFrame().reindex_like(bench_pts)
        rolled = pd.DataFrame().reindex_like(bench_pts)
        rolled = rolled.fillna(0)

        free_transfer = np.zeros(105000)

        for gw in np.arange(1, 39):
            for player in hit_points.index:
                # Not registered team
                if not len(teams.loc[player, str(gw)]):
                    hit_points.loc[player, str(gw)] = 0
                    free_transfers.loc[player, str(gw)] = 1
                    continue

                # Hits
                try:
                    if not (chips.loc[int(player), 'freehit'] == gw or chips.loc[int(player), 'wildcard_1'] == gw or chips.loc[int(player), 'wildcard_2'] == gw) :
                        transfer = 0 if transfers.loc[player, str(gw)] != transfers.loc[player, str(gw)] else len(transfers.loc[player, str(gw)]['in'])
                        hit = free_transfer[int(player)] - transfer
                        hit_points.loc[player, str(gw)] = 0
                        if hit > 0:
                            # No transfer
                            free_transfer[int(player)] = 2
                            rolled.loc[int(player), str(gw)] = 1
                        elif hit == 0 :
                            # Used all FT
                            free_transfer[int(player)] = 1
                        else:
                            # Hits
                            free_transfer[int(player)] = 1
                            hit_points.loc[player, str(gw)] = hit

                    else:
                        free_transfer[int(player)] = 1
                except:
                    free_transfer[int(player)] = min(2, free_transfer[int(player)] + 1)
                free_transfers.loc[player, str(gw)] = free_transfer[int(player)]

        hit_points = hit_points.fillna(0)
        hit_points = hit_points.astype(int)
        free_transfers = free_transfers.fillna(0)
        free_transfers = free_transfers.astype(int)
        rolled = rolled.fillna(0)
        rolled = rolled.astype(int)

        hit_points.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/hit_points_{rank}.csv')
        free_transfers.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/free_transfers_{rank}.csv')
        rolled.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/rolled_{rank}.csv')


def get_season_misc():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    all_gw_data = pd.read_csv(os.path.join(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/gws/merged_gw.csv'))[['GW', 'position', 'element', 'minutes']]
    mapping = {
        'DEF': 1,
        'MID': 2,
        'FWD': 3,
        'GK': 4
    }

    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank)

        transfer_position = pd.DataFrame().reindex_like(bench_pts)
        captain_position = pd.DataFrame().reindex_like(bench_pts)

        for gw in np.arange(1, 39):
            gw_data = all_gw_data[all_gw_data['GW'] == gw][['position', 'element', 'minutes']]

            for player in captain_position.index:
                # Not registered team
                if not len(teams.loc[player, str(gw)]):
                    transfer_position.loc[player, str(gw)] = 0
                    captain_position.loc[player, str(gw)] = 0
                    continue

                # Captain/Vice points
                try:
                    # This lookup throws an exception if the player has a BGW
                    if sum(gw_data[gw_data['element'] == caps.loc[player, str(gw)]]['minutes']) == 0:
                        # Captain does not play
                        captain_position.loc[player, str(gw)] = mapping[gw_data[gw_data['element'] == vice.loc[player, str(gw)]]['position'].values[0]]
                    else:
                        captain_position.loc[player, str(gw)] = mapping[gw_data[gw_data['element'] == caps.loc[player, str(gw)]]['position'].values[0]]
                except:
                    try:
                        captain_position.loc[player, str(gw)] = mapping[gw_data[gw_data['element'] == vice.loc[player, str(gw)]]['position'].values[0]]
                    except:
                        # Neither vice or cap plays :(
                        captain_position.loc[player, str(gw)] = 0

                # Hits
                try:
                    if not (chips.loc[int(player), 'freehit'] == gw or chips.loc[int(player), 'wildcard_1'] == gw or chips.loc[int(player), 'wildcard_2'] == gw) :
                        if len(transfers.loc[player, str(gw)]['in']) > 0 :
                            transfer_position.loc[player, str(gw)] = 0
                            tran = list(gw_data[gw_data['element'].isin(list(transfers.loc[player, str(gw)]['in'].values()))].drop_duplicates(subset='element', keep="first")['position'].values)
                            for idx, _ in enumerate(transfers.loc[player, str(gw)]['in']):
                                transfer_position.loc[player, str(gw)] += np.power(10, idx) * mapping[tran[idx]]

                    else:
                        transfer_position.loc[player, str(gw)] = 0
                except:
                    transfer_position.loc[player, str(gw)] = 0

        captain_position = captain_position.fillna(0)
        captain_position = captain_position.astype(int)
        transfer_position = transfer_position.fillna(0)
        transfer_position = transfer_position.astype(int)

        captain_position.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/captain_position_{rank}.csv')
        transfer_position.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/processed/transfer_position_{rank}.csv')


def get_hof_transfers():
    with open('info.json') as f:
        season_data = json.load(f)
    season = season_data['season']

    chips, teams, caps, vice, bench_pts, transfers = get_raw_data('hof', f'data/fpl_official/{season}-{season % 2000 + 1}/season/')

    hit_points = pd.DataFrame().reindex_like(bench_pts)
    free_transfers = pd.DataFrame().reindex_like(bench_pts)
    free_transfer = pd.DataFrame().reindex_like(bench_pts)[['1']]
    free_transfer = free_transfer.fillna(0)
    rolled = pd.DataFrame().reindex_like(bench_pts)
    rolled = rolled.fillna(0)

    for gw in np.arange(1, 39):
        for player in hit_points.index:

            # Not registered team
            if not len(teams.loc[player, str(gw)]):
                hit_points.loc[player, str(gw)] = 0
                free_transfers.loc[player, str(gw)] = 1
                rolled.loc[player, str(gw)] = 0
                continue

            # Hits
            try:
                if not (chips.loc[int(player), 'freehit'] == gw or chips.loc[int(player), 'wildcard_1'] == gw or chips.loc[int(player), 'wildcard_2'] == gw) :
                    transfer = 0 if transfers.loc[player, str(gw)] != transfers.loc[player, str(gw)] else len(transfers.loc[player, str(gw)]['in'])
                    hit = free_transfer.loc[int(player), '1'] - transfer
                    hit_points.loc[player, str(gw)] = 0
                    if hit > 0:
                        # No transfer
                        free_transfer.loc[int(player), '1'] = 2
                        rolled.loc[int(player), str(gw)] = 1
                    elif hit == 0 :
                        # Used all FT
                        free_transfer.loc[int(player), '1'] = 1
                    else:
                        # Hits
                        free_transfer.loc[int(player), '1'] = 1
                        hit_points.loc[player, str(gw)] = hit
    
                else:
                    free_transfer.loc[int(player), '1'] = 1
            except:
                free_transfer.loc[int(player), '1'] = min(2, free_transfer.loc[int(player), '1'] + 1)
            free_transfers.loc[player, str(gw)] = free_transfer.loc[int(player), '1']

    hit_points = hit_points.fillna(0)
    hit_points = hit_points.astype(int)
    free_transfers = free_transfers.fillna(0)
    free_transfers = free_transfers.astype(int)
    rolled = rolled.fillna(0)
    rolled = rolled.astype(int)

    hit_points.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/hit_points_hof.csv')
    free_transfers.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/free_transfers_hof.csv')
    rolled.to_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/season/rolled_hof.csv')


# get_season_points()
# get_season_value()
# get_season_formation()
# get_season_pos_values()
get_season_assets()
# get_season_transfers()
# get_season_misc()
# get_hof_transfers()
