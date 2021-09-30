import pandas as pd
import os
import sasoptpy as so


# Using the predicted points from https://fplreview.com/
filepath = "../data/fpl_review/2021-22/gameweek/7/fplreview_fp.csv"
# Gameweek of interest
gw = '7_Pts'
budget = 1000
# Get the goalkeeper data
df = pd.read_csv(filepath)
df["Pos"] = df["Pos"].map(
    {
        1: 'G',
        2: 'D',
        3: 'M',
        4: 'F'
        })
df = pd.concat([df, pd.get_dummies(df.Pos)], axis=1)
df = pd.concat([df, pd.get_dummies(df.Team)], axis=1)
data = df.copy().reset_index()
data.set_index('index', inplace=True)
players = data.index.tolist()

# Create a model
model = so.Model(name='single_gw_model')

# Define variables
lineup = model.add_variables(players, name='lineup', vartype=so.binary)
bench = model.add_variables(players, name='bench', vartype=so.binary)
captain = model.add_variables(players, name='captain', vartype=so.binary)
vicecaptain = model.add_variables(players, name='vicecaptain', vartype=so.binary)

# Define Objective: maximize total expected points
# Assume a 10% chance of a player not playing
total_xp_lineup = so.expr_sum(lineup[p] * data.loc[p, gw] for p in players)
total_xp_bench = 0.1 * so.expr_sum(bench[p] * data.loc[p, gw] for p in players)
total_xp_captain = so.expr_sum(captain[p] * data.loc[p, gw] for p in players)
total_xp_vicecaptain = 0.1 * so.expr_sum(vicecaptain[p] * data.loc[p, gw] for p in players)
model.set_objective(-total_xp_lineup - total_xp_bench - total_xp_captain - total_xp_vicecaptain, name='total_xp_obj', sense='N')

# Define Constraints
# The cost of the squad must exceed the budget
model.add_constraint(so.expr_sum((lineup[p] + bench[p]) * data.loc[p, 'BV'] for p in players) <= budget,
                     name='budget')

# The number of keeper must be 11 on field and 4 on bench
model.add_constraint(so.expr_sum(lineup[p] for p in players) == 11, name='11_starters')
model.add_constraint(so.expr_sum(bench[p] for p in players) == 4, name='4_substitutes')
model.add_constraint(so.expr_sum(captain[p] for p in players) == 1, name='1_captain')
model.add_constraint(so.expr_sum(vicecaptain[p] for p in players) == 1, name='1_vicecaptain')

# A player, captain and vicecaptain must not be picked more than once
model.add_constraints((lineup[p] + bench[p] <= 1 for p in players), name='lineup_or_bench')
model.add_constraints((captain[p] + vicecaptain[p] <= 1 for p in players), name='cap_or_vicecap')

# The number of players from a team must not be more than three
team_name = df.columns[-20:].values
model.add_constraints((so.expr_sum((lineup[p] + bench[p]) *
                                   data.loc[p, team] for p in players) <= 3
                       for team in team_name), name='team_limit')

# The number of players fit the requirements 2 Gk, 5 Def, 5 Mid, 3 For
model.add_constraint(so.expr_sum((lineup[p] + bench[p]) * data.loc[p, 'G'] for p in players) == 2, name='gk_limit')
model.add_constraint(so.expr_sum((lineup[p] + bench[p]) * data.loc[p, 'D'] for p in players) == 5, name='def_limit')
model.add_constraint(so.expr_sum((lineup[p] + bench[p]) * data.loc[p, 'M'] for p in players) == 5, name='mid_limit')
model.add_constraint(so.expr_sum((lineup[p] + bench[p]) * data.loc[p, 'F'] for p in players) == 3, name='for_limit')

# The formation is valid i.e. Minimum one goalkeeper, 3 defenders, 2 midfielders and 1 striker on the lineup
model.add_constraint(so.expr_sum(lineup[p] * data.loc[p, 'G'] for p in players) == 1, name='gk_min')
model.add_constraint(so.expr_sum(lineup[p] * data.loc[p, 'D'] for p in players) >= 3, name='def_min')
model.add_constraint(so.expr_sum(lineup[p] * data.loc[p, 'M'] for p in players) >= 2, name='mid_min')
model.add_constraint(so.expr_sum(lineup[p] * data.loc[p, 'F'] for p in players) >= 1, name='for_min')

# The captain and vicecaptain must be a player on the field
model.add_constraints((captain[p] <= lineup[p] for p in players), name='captain_lineup_rel')
model.add_constraints((vicecaptain[p] <= lineup[p] for p in players), name='vicecap_lineup_rel')

# Solve Step
model.export_mps(filename='single_gw.mps')
command = 'cbc single_gw.mps solve solu single_gw_solution.txt'
# !{command}
os.system(command)
with open('single_gw_solution.txt', 'r') as f:
    for v in model.get_variables():
        v.set_value(0)
    for line in f:
        if 'objective value' in line:
            continue
        words = line.split()
        var = model.get_variable(words[1])
        var.set_value(float(words[2]))

squad_value = bench_value = squad_xp = 0

team = pd.DataFrame([], columns=['Name', 'Pos', 'Team', 'BV', 'xP', 'Start', 'Captain', 'Vicecaptain'])

# GK
for p in players:
    if lineup[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'G':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw]

# DEF
for p in players:
    if lineup[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'D':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw]

# MID
for p in players:
    if lineup[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'M':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw]

# FOR
for p in players:
    if lineup[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'F':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw]

# CAP
for p in players:
    if captain[p].get_value() > 0.5:
        team.loc[team['Name'] == data.loc[p]['Name'], ['Captain']] = 1
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_xp += data.loc[p][gw]

# VICECAP
for p in players:
    if vicecaptain[p].get_value() > 0.5:
        team.loc[team['Name'] == data.loc[p]['Name'], ['Vicecaptain']] = 1
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])

# BENCH
for p in players:
    if bench[p].get_value() > 0.5:
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw], 'Start': 0, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        bench_value += data.loc[p].BV
        squad_xp += 0.1 * data.loc[p][gw]

print('Squad value: {} | Bench value: {} | XP: {}'.format(squad_value, bench_value, squad_xp))

print(team)
