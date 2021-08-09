import pandas as pd
import numpy as np
import glob
import os

def get_raw_data(rank):
    path = '../data/fpl_official/20-21/season/raw/'
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
    for rank in np.arange(5000, 300000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank)

        points = pd.DataFrame().reindex_like(bench_pts)

        free_transfer = np.zeros(300000)

        all_gw_data = pd.read_csv(os.path.join('../data/fpl_official/vaastav/data/2020-21/gws/merged_gw.csv'))[['GW', 'position', 'element', 'total_points', 'minutes', 'value']]

        for gw in np.arange(1, 39):
            print(gw)
            gw_data = all_gw_data[all_gw_data['GW'] == gw]
            next_gw_data = all_gw_data[all_gw_data['GW'] == gw+1]

            for player in points.index:
                # Not registered team
                if not len(teams.loc[player, str(gw)]):
                    points.loc[player, str(gw)] = 0
                    continue

                # FPL Team
                fpl_team = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)])]
                fpl_bench = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)][-3:])]

                # All player pts
                points.loc[player, str(gw)] = sum(fpl_team['total_points'])

                fpl_team = fpl_team.drop_duplicates(subset='element', keep="first")
                fpl_bench = fpl_bench.drop_duplicates(subset='element', keep="first")

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
                            points.loc[int(player/rank), str(gw)] += 4 * hit
                            free_transfer[int(player)] = 1

                    else:
                        free_transfer[player] = 1
                except:
                    free_transfer[int(player)] = min(2, free_transfer[int(player)] + 1)

        points = points.fillna(0)
        points = points.astype(int)
        points.to_csv(f'../data/fpl_official/20-21/season/processed/points_{rank}.csv')


def get_season_value():
    for rank in np.arange(5000, 105000, 5000):
        print(rank)
        chips, teams, caps, vice, bench_pts, transfers = get_raw_data(rank)

        team_value = pd.DataFrame().reindex_like(bench_pts)
        in_the_bank = pd.DataFrame().reindex_like(bench_pts)
        bench_value = pd.DataFrame().reindex_like(bench_pts)

        free_transfer = np.zeros(300000)

        all_gw_data = pd.read_csv(os.path.join('../data/fpl_official/vaastav/data/2020-21/gws/merged_gw.csv'))[['GW', 'position', 'element', 'total_points', 'minutes', 'value']]

        for gw in np.arange(1, 39):
            print(gw)
            gw_data = all_gw_data[all_gw_data['GW'] == gw]
            next_gw_data = all_gw_data[all_gw_data['GW'] == gw+1]

            for player in team_value.index:
                # Not registered team
                if not len(teams.loc[player, str(gw)]):
                    team_value.loc[player, str(gw)] = 1000
                    continue

                # FPL Team
                fpl_team = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)])]
                fpl_bench = gw_data[gw_data['element'].isin(teams.loc[player, str(gw)][-3:])]
                next_fpl_team = next_gw_data[next_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")
                next_fpl_bench = next_gw_data[next_gw_data['element'].isin([player_id for player_id in teams.loc[player, str(gw)][-3:] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")
                fpl_team = fpl_team.drop_duplicates(subset='element', keep="first")
                fpl_bench = fpl_bench.drop_duplicates(subset='element', keep="first")

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
                            points.loc[int(player/rank), str(gw)] += 4 * hit
                            free_transfer[int(player)] = 1

                    else:
                        free_transfer[player] = 1
                except:
                    free_transfer[int(player)] = min(2, free_transfer[int(player)] + 1)

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
                        in_the_bank.loc[player, str(gw)] +=  max(0,
                            sum(gw_data[gw_data['element'].isin(list(transfers.loc[player, str(gw)]['out'].values()))].drop_duplicates(subset='element', keep="first")['value']) +
                            sum(next_gw_data[next_gw_data['element'].isin([player_id for player_id in transfers.loc[player, str(gw)]['out'] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")['value']) -
                            sum(gw_data[gw_data['element'].isin(list(transfers.loc[player, str(gw)]['in'].values()))].drop_duplicates(subset='element', keep="first")['value']) -
                            sum(next_gw_data[next_gw_data['element'].isin([player_id for player_id in transfers.loc[player, str(gw)]['in'] if player_id not in gw_data['element'].values])].drop_duplicates(subset='element', keep="first")['value'])
                            )
                else:
                    in_the_bank.loc[player, str(gw)] = in_the_bank.loc[player, str(gw-1)]

        team_value = team_value.fillna(0)
        team_value = team_value.astype(int)
        in_the_bank = in_the_bank.fillna(0)
        in_the_bank = in_the_bank.astype(int)
        bench_value = bench_value.fillna(0)
        bench_value = bench_value.astype(int)

        team_value.to_csv(f'../data/fpl_official/20-21/season/processed/team_value_{rank}.csv')
        in_the_bank.to_csv(f'../data/fpl_official/20-21/season/processed/in_the_bank_{rank}.csv')
        bench_value.to_csv(f'../data/fpl_official/20-21/season/processed/bench_value_{rank}.csv')

# get_season_points()
get_season_value()