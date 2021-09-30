import pandas as pd
import os
import sasoptpy as so


# Using the predicted points from https://fplreview.com/
filepath = "../data/fpl_review/2021-22/gameweek/7/fplreview_fp.csv"
# User {Hyper}parameters
base = 7
gw_t1 = str(base) + "_Pts"
gw_t2 = str(base + 1) + "_Pts"
budget = 1000
free_transfers = 1
decay_bench = 0.1
decay_gameweek = 0.8
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
model = so.Model(name='gameweek_model')

# Define variables
team_t = model.add_variables(players, name='players_in_common', vartype=so.binary)

team_t1 = model.add_variables(players, name='team_t1', vartype=so.binary)
starter_t1 = model.add_variables(players, name='starter_t1', vartype=so.binary)
captain_t1 = model.add_variables(players, name='captain_t1', vartype=so.binary)
vicecaptain_t1 = model.add_variables(players, name='vicecaptain_t1', vartype=so.binary)

team_t2 = model.add_variables(players, name='team_t2', vartype=so.binary)
starter_t2 = model.add_variables(players, name='starter_t2', vartype=so.binary)
captain_t2 = model.add_variables(players, name='captain_t2', vartype=so.binary)
vicecaptain_t2 = model.add_variables(players, name='vicecaptain_t2', vartype=so.binary)

# Define Objective: maximize total expected points
# Assume a 10% (decay_bench) chance of a player not playing
# Assume a 80% (decay_gameweek) reliability of next week's xPts
xp_t1 = so.expr_sum((starter_t1[p] + captain_t1[p] + decay_bench * (vicecaptain_t1[p] + team_t1[p] - starter_t1[p])) *
                    data.loc[p, gw_t1] for p in players)
xp_t2 = so.expr_sum((starter_t2[p] + captain_t2[p] + decay_bench * (vicecaptain_t2[p] + team_t2[p] - starter_t2[p])) *
                    data.loc[p, gw_t2] for p in players)

model.set_objective(- xp_t1 - decay_gameweek * xp_t2, name='total_xp_obj', sense='N')

# Define Constraints
# The cost of the squad must exceed the budget
model.add_constraint(so.expr_sum(team_t1[p] * data.loc[p, 'BV'] for p in players) <= budget, name='budget_t1')
model.add_constraint(so.expr_sum(team_t2[p] * data.loc[p, 'BV'] for p in players) <= budget, name='budget_t2')

# The number of keeper must be 11 on field and 4 on bench
model.add_constraint(so.expr_sum(team_t1[p] for p in players) == 15, name='15_starters_t1')
model.add_constraint(so.expr_sum(starter_t1[p] for p in players) == 11, name='11_starters_t1')
model.add_constraint(so.expr_sum(captain_t1[p] for p in players) == 1, name='1_captain_t1')
model.add_constraint(so.expr_sum(vicecaptain_t1[p] for p in players) == 1, name='1_vicecaptain_t1')

model.add_constraint(so.expr_sum(team_t2[p] for p in players) == 15, name='15_starters_t2')
model.add_constraint(so.expr_sum(starter_t2[p] for p in players) == 11, name='11_starters_t2')
model.add_constraint(so.expr_sum(captain_t2[p] for p in players) == 1, name='1_captain_t2')
model.add_constraint(so.expr_sum(vicecaptain_t2[p] for p in players) == 1, name='1_vicecaptain_t2')

# A captain must not be picked more than once
model.add_constraints((captain_t1[p] + vicecaptain_t1[p] <= 1 for p in players), name='cap_or_vice_t1')
model.add_constraints((captain_t2[p] + vicecaptain_t2[p] <= 1 for p in players), name='cap_or_vice_t2')

# The number of players from a team must not be more than three
team_name = df.columns[-20:].values
model.add_constraints((so.expr_sum(team_t1[p] * data.loc[p, team] for p in players) <= 3
                       for team in team_name), name='team_limit_t1')

model.add_constraints((so.expr_sum(team_t2[p] * data.loc[p, team] for p in players) <= 3
                       for team in team_name), name='team_limit_t2')

# The number of players fit the requirements 2 Gk, 5 Def, 5 Mid, 3 For
model.add_constraint(so.expr_sum(team_t1[p] * data.loc[p, 'G'] for p in players) == 2, name='gk_limit_t1')
model.add_constraint(so.expr_sum(team_t1[p] * data.loc[p, 'D'] for p in players) == 5, name='def_limit_t1')
model.add_constraint(so.expr_sum(team_t1[p] * data.loc[p, 'M'] for p in players) == 5, name='mid_limit_t1')
model.add_constraint(so.expr_sum(team_t1[p] * data.loc[p, 'F'] for p in players) == 3, name='for_limit_t1')

model.add_constraint(so.expr_sum(team_t2[p] * data.loc[p, 'G'] for p in players) == 2, name='gk_limit_t2')
model.add_constraint(so.expr_sum(team_t2[p] * data.loc[p, 'D'] for p in players) == 5, name='def_limit_t2')
model.add_constraint(so.expr_sum(team_t2[p] * data.loc[p, 'M'] for p in players) == 5, name='mid_limit_t2')
model.add_constraint(so.expr_sum(team_t2[p] * data.loc[p, 'F'] for p in players) == 3, name='for_limit_t2')

# The formation is valid i.e. Minimum one goalkeeper, 3 defenders, 2 midfielders and 1 striker on the lineup
model.add_constraint(so.expr_sum(starter_t1[p] * data.loc[p, 'G'] for p in players) == 1, name='gk_min_t1')
model.add_constraint(so.expr_sum(starter_t1[p] * data.loc[p, 'D'] for p in players) >= 3, name='def_min_t1')
model.add_constraint(so.expr_sum(starter_t1[p] * data.loc[p, 'M'] for p in players) >= 2, name='mid_min_t1')
model.add_constraint(so.expr_sum(starter_t1[p] * data.loc[p, 'F'] for p in players) >= 1, name='for_min_t1')

model.add_constraint(so.expr_sum(starter_t2[p] * data.loc[p, 'G'] for p in players) == 1, name='gk_min_t2')
model.add_constraint(so.expr_sum(starter_t2[p] * data.loc[p, 'D'] for p in players) >= 3, name='def_min_t2')
model.add_constraint(so.expr_sum(starter_t2[p] * data.loc[p, 'M'] for p in players) >= 2, name='mid_min_t2')
model.add_constraint(so.expr_sum(starter_t2[p] * data.loc[p, 'F'] for p in players) >= 1, name='for_min_t2')

# The captain & vicecap must be a player on the field
model.add_constraints((captain_t1[p] <= starter_t1[p] for p in players), name='captain_in_starters_t1')
model.add_constraints((vicecaptain_t1[p] <= starter_t1[p] for p in players), name='vicecaptain_in_starters_t1')

model.add_constraints((captain_t2[p] <= starter_t2[p] for p in players), name='captain_in_starters_t2')
model.add_constraints((vicecaptain_t2[p] <= starter_t2[p] for p in players), name='vicecaptain_in_starters_t2')

# The starters must be in the team
model.add_constraints((starter_t1[p] <= team_t1[p] for p in players), name='starters_in_team_t1')

model.add_constraints((starter_t2[p] <= team_t2[p] for p in players), name='starters_in_team_t2')

# The teams must have players in common (15 - the number of FT)
model.add_constraint((so.expr_sum(team_t[p] for p in players) == 15 - free_transfers), name='players_in_common')
model.add_constraints((team_t[p] >= team_t1[p] + team_t2[p] - 1 for p in players), name='auxiliary_var')
model.add_constraints((team_t[p] <= team_t1[p] for p in players), name='t1_in_t')
model.add_constraints((team_t[p] <= team_t2[p] for p in players), name='t2_in_t')

# Solve Step
model.export_mps(filename='gameweek.mps')
command = 'cbc gameweek.mps solve solu gameweek_solution.txt'
# !{command}
os.system(command)
with open('gameweek_solution.txt', 'r') as f:
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
    if starter_t1[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'G':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t1], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw_t1]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t1]

# DEF
for p in players:
    if starter_t1[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'D':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t1], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t1]

# MID
for p in players:
    if starter_t1[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'M':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t1], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t1]

# FOR
for p in players:
    if starter_t1[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'F':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t1], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t1]

# CAP
for p in players:
    if captain_t1[p].get_value() > 0.5:
        team.loc[team['Name'] == data.loc[p]['Name'], ['Captain']] = 1
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_xp += data.loc[p][gw_t1]

# VICECAP
for p in players:
    if vicecaptain_t1[p].get_value() > 0.5:
        team.loc[team['Name'] == data.loc[p]['Name'], ['Vicecaptain']] = 1
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_xp += data.loc[p][gw_t1] * 0.1

# BENCH
for p in players:
    if team_t1[p].get_value() > 0.5 and starter_t1[p].get_value() == 0:
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t1], 'Start': 0, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        bench_value += data.loc[p].BV
        squad_xp += 0.1 * data.loc[p][gw_t1]

print('Squad value: {} | Bench value: {} | XP: {}'.format(squad_value, bench_value, squad_xp))
print(team)

squad_value = bench_value = squad_xp = 0
team = pd.DataFrame([], columns=['Name', 'Pos', 'Team', 'BV', 'xP', 'Start', 'Captain', 'Vicecaptain'])

# GK
for p in players:
    if starter_t2[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'G':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t2], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw_t1]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t2]

# DEF
for p in players:
    if starter_t2[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'D':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t2], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t2]

# MID
for p in players:
    if starter_t2[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'M':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t2], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t2]

# FOR
for p in players:
    if starter_t2[p].get_value() > 0.5 and data.loc[p]['Pos'] == 'F':
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t2], 'Start': 1, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_value += data.loc[p].BV
        squad_xp += data.loc[p][gw_t2]

# CAP
for p in players:
    if captain_t2[p].get_value() > 0.5:
        team.loc[team['Name'] == data.loc[p]['Name'], ['Captain']] = 1
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_xp += data.loc[p][gw_t2]

# VICECAP
for p in players:
    if vicecaptain_t2[p].get_value() > 0.5:
        team.loc[team['Name'] == data.loc[p]['Name'], ['Vicecaptain']] = 1
        # print(p, data.loc[p][['Name', 'BV', 'Team', gw]])
        squad_xp += data.loc[p][gw_t2] * 0.1

# BENCH
for p in players:
    if team_t2[p].get_value() > 0.5 and starter_t2[p].get_value() == 0:
        team = team.append({'Name': data.loc[p]['Name'], 'Pos': data.loc[p]['Pos'], 'Team': data.loc[p]['Team'],
                            'BV': data.loc[p]['BV'], 'xP': data.loc[p][gw_t2], 'Start': 0, 'Captain': 0, 'Vicecaptain': 0},
                           ignore_index=True)
        bench_value += data.loc[p].BV
        squad_xp += 0.1 * data.loc[p][gw_t2]

print('Squad value: {} | Bench value: {} | XP: {}'.format(squad_value, bench_value, squad_xp))
print(team)