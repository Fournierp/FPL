import pandas as pd
import numpy as np
import os
import sasoptpy as so
from utils import get_team, get_predictions, get_rolling, pretty_print

# User {Hyper}parameters
start = 7
decay_bench = 0.1
decay_gameweek = 0.8
team_id = 35868

# Data collection
# Predicted points from https://fplreview.com/
df = get_predictions()
data = df.copy()
data.set_index('id', inplace=True)
players = data.index.tolist()

# FPL data
initial_team, bank = get_team(team_id, start - 1)

# GW
period = min(5, len([col for col in df.columns if '_Pts' in col]))
rolling_transfer, transfer = get_rolling(team_id, start - 1) 
budget = np.sum([data.loc[p, 'SV'] for p in initial_team]) + bank
all_gameweeks = np.arange(start-1, start+period)
gameweeks = np.arange(start, start+period)

# Model
model = so.Model(name='season_model')

# Variables
team = model.add_variables(players, all_gameweeks, name='team', vartype=so.binary)
starter = model.add_variables(players, gameweeks, name='starter', vartype=so.binary)
captain = model.add_variables(players, gameweeks, name='captain', vartype=so.binary)
vicecaptain = model.add_variables(players, gameweeks, name='vicecaptain', vartype=so.binary)

buy = model.add_variables(players, gameweeks, name='buy', vartype=so.binary)
sell = model.add_variables(players, gameweeks, name='sell', vartype=so.binary)

free_transfers = model.add_variables(all_gameweeks, name='ft', vartype=so.integer, lb=1, ub=2)
hits = model.add_variables(gameweeks, name='hits', vartype=so.integer, lb=0, ub=15)
rolling_transfers = model.add_variables(gameweeks, name='rolling', vartype=so.binary)


# Objective: maximize total expected points
# Assume a 10% (decay_bench) chance of a player not playing
# Assume a 80% (decay_gameweek) reliability of next week's xPts
xp = so.expr_sum(
    np.power(decay_gameweek, w - start) *
    so.expr_sum((starter[p, w] + captain[p, w] + decay_bench * (vicecaptain[p, w] + team[p, w] - starter[p, w])) *
                data.loc[p, str(w) + '_Pts'] for p in players) -
    4 * hits[w] for w in gameweeks)

model.set_objective(- xp, name='total_xp_obj', sense='N')

# Initial conditions: set team and FT depending on the team
model.add_constraints((team[p, start - 1] == 1 for p in initial_team), name='initial_team')
model.add_constraint(free_transfers[start - 1] == rolling_transfer + 1, name='initial_ft')


# Constraints
# The cost of the squad must exceed the budget
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, 'BV'] for p in players) <= budget for w in all_gameweeks), name='budget')

# The number of players must be 11 on field, 4 on bench, 1 captain & 1 vicecaptain
model.add_constraints((so.expr_sum(team[p, w] for p in players) == 15 for w in all_gameweeks), name='15_players')
model.add_constraints((so.expr_sum(starter[p, w] for p in players) == 11 for w in gameweeks), name='11_starters')
model.add_constraints((so.expr_sum(captain[p, w] for p in players) == 1 for w in gameweeks), name='1_captain')
model.add_constraints((so.expr_sum(vicecaptain[p, w] for p in players) == 1 for w in gameweeks), name='1_vicecaptain')

# A captain must not be picked more than once
model.add_constraints((captain[p, w] + vicecaptain[p, w] <= 1 for p in players for w in gameweeks), name='cap_or_vice')

# The number of players from a team must not be more than three
team_names = df.columns[-20:].values
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, team_name] for p in players) <= 3
                       for team_name in team_names for w in gameweeks), name='team_limit')

# The number of players fit the requirements 2 Gk, 5 Def, 5 Mid, 3 For
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, 'G'] for p in players) == 2 for w in gameweeks), name='gk_limit')
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, 'D'] for p in players) == 5 for w in gameweeks), name='def_limit')
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, 'M'] for p in players) == 5 for w in gameweeks), name='mid_limit')
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, 'F'] for p in players) == 3 for w in gameweeks), name='for_limit')

# The formation is valid i.e. Minimum one goalkeeper, 3 defenders, 2 midfielders and 1 striker on the lineup
model.add_constraints((so.expr_sum(starter[p, w] * data.loc[p, 'G'] for p in players) == 1 for w in gameweeks), name='gk_min')
model.add_constraints((so.expr_sum(starter[p, w] * data.loc[p, 'D'] for p in players) >= 3 for w in gameweeks), name='def_min')
model.add_constraints((so.expr_sum(starter[p, w] * data.loc[p, 'M'] for p in players) >= 2 for w in gameweeks), name='mid_min')
model.add_constraints((so.expr_sum(starter[p, w] * data.loc[p, 'F'] for p in players) >= 1 for w in gameweeks), name='for_min')

# The captain & vicecap must be a player on the field
model.add_constraints((captain[p, w] <= starter[p, w] for p in players for w in gameweeks), name='captain_in_starters')
model.add_constraints((vicecaptain[p, w] <= starter[p, w] for p in players for w in gameweeks), name='vicecaptain_in_starters')

# The starters must be in the team
model.add_constraints((starter[p, w] <= team[p, w] for p in players for w in gameweeks), name='starters_in_team')

# The team must be equal to the next week excluding transfers
model.add_constraints((team[p, w] == team[p, w - 1] + buy[p, w] - sell[p, w] for p in players for w in gameweeks),
                      name='team_transfer')

# The rolling transfer must be equal to the number of free transfers not used (+ 1)
model.add_constraints((free_transfers[w] == rolling_transfers[w] + 1 for w in gameweeks), name='rolling_ft_rel')

# Rolling transfers
number_of_transfers = {w: so.expr_sum(sell[p, w] for p in players) for w in gameweeks}
number_of_transfers[start - 1] = transfer
model.add_constraints((free_transfers[w - 1] - number_of_transfers[w - 1] <= 2 * rolling_transfers[w] for w in gameweeks),
                      name='rolling_condition_1')
model.add_constraints(
    (free_transfers[w - 1] - number_of_transfers[w - 1] >= rolling_transfers[w] + (-14) * (1 - rolling_transfers[w])
     for w in gameweeks),
    name='rolling_condition_2')

# The number of hits must be the number of transfer except the free ones.
model.add_constraints((hits[w] >= number_of_transfers[w] - free_transfers[w] for w in gameweeks), name='hits')


# Solve Step
model.export_mps(filename='season.mps')
command = 'cbc season.mps solve solu season_solution.txt'
# !{command}
os.system(command)
with open('season_solution.txt', 'r') as f:
    for v in model.get_variables():
        v.set_value(0)
    for line in f:
        if 'objective value' in line:
            continue
        words = line.split()
        var = model.get_variable(words[1])
        var.set_value(float(words[2]))

# GW
pretty_print(data, start, period, team, starter, captain, vicecaptain, buy, sell, free_transfers, hits)
