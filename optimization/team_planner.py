import pandas as pd
import numpy as np
import os
from subprocess import Popen, DEVNULL
import sasoptpy as so
from utils import (
    get_team,
    get_predictions,
    get_rolling,
    pretty_print,
    get_chips,
    get_next_gw,
    get_ownership_data)

# User {Hyper}parameters
decay_bench = 0.05
decay_gameweek = 0.8
team_id = 35868
horizon = 5
nb_suboptimal = 3
cutoff_search = 'first_transfer'
objective_type = 'decay'
# Chip strategy: set to (-1) if you don't want to use
# Choose a value in range [0-5] as the number of gameweeks after the current one
freehit_gw = -1
wildcard_gw = -1
bboost_gw = -1
threexc_gw = -1

assert (freehit_gw < 5), "Select a gameweek within the horizon."
assert (wildcard_gw < 5), "Select a gameweek within the horizon."
assert (bboost_gw < 5), "Select a gameweek within the horizon."
assert (threexc_gw < 5), "Select a gameweek within the horizon."

# Biased decisions for player choices 
# Example usage: {(index, gw)}
love = {
    'buy': {},
    'start': {},
    'team': {},
    'cap': {}
}

hate = {
    'sell': {},
    'team': {},
    'bench': {}
}
# Biased decisions for transfter limits (gw, limit)
# Example usage: (gw, amount)
hit_limit = {
    'max': {},
    'eq': {},
    'min': {}
}
# Gameweeks where 2FT are desired. In other words the (GW-1) where a FT is rolled.
two_ft_gw = []
# Differential strategy
nb_differentials = 3
differential_threshold = 10

log = True #False

# Data collection
# Predicted points from https://fplreview.com/
df = get_predictions(noise=False)
data = df.copy()
data.set_index('id', inplace=True)

# Ownership data
ownership = get_ownership_data()
data = pd.concat([data, ownership], axis=1, join="inner")
players = data.index.tolist()

# FPL data
start = get_next_gw()
initial_team, bank = get_team(team_id, start - 1)
freehit_used, wildcard_used, bboost_used, threexc_used = get_chips(team_id, start - 1)

# GW
period = min(horizon, len([col for col in df.columns if '_Pts' in col]))
rolling_transfer, transfer = get_rolling(team_id, start - 1) 
budget = np.sum([data.loc[p, 'SV'] for p in initial_team]) + bank
all_gameweeks = np.arange(start-1, start+period)
gameweeks = np.arange(start, start+period)

assert not (freehit_used and freehit_gw >= 0), "Freehit chip was already used."
assert not (wildcard_used and wildcard_gw >= 0), "Wildcard chip was already used."
assert not (bboost_used and bboost_gw >= 0), "Bench boost chip was already used."
assert not (threexc_used and threexc_gw >= 0), "Tripple captain chip was already used."

# Sort DF by EV for efficient optimization
data['total_ev'] = data[[col for col in df.columns if '_Pts' in col]].sum(axis=1)
data.sort_values(by=['total_ev'], ascending=[False], inplace=True)


# Model
model_name = 'suboptimal'
model = so.Model(name=model_name + '_model')

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

freehit = model.add_variables(gameweeks, name='fh', vartype=so.integer, lb=0, ub=15)
wildcard = model.add_variables(gameweeks, name='wc', vartype=so.integer, lb=0, ub=15)
bboost = model.add_variables(gameweeks, name='bb', vartype=so.binary)
threexc = model.add_variables(players, gameweeks, name='3xc', vartype=so.binary)

force_in = model.add_variables(players, gameweeks, name='fi', vartype=so.binary)
force_out = model.add_variables(players, gameweeks, name='fo', vartype=so.binary)


# Objective: maximize total expected points
# Assume a 10% (decay_bench) chance of a player not playing
# Assume a 80% (decay_gameweek) reliability of next week's xPts
xp = so.expr_sum(
    (np.power(decay_gameweek, w - start) if objective_type == 'linear' else 1) *
    (
            so.expr_sum(
                (starter[p, w] + captain[p, w] + threexc[p, w] +
                 decay_bench * (vicecaptain[p, w] + team[p, w] - starter[p, w])) *
                data.loc[p, str(w) + '_Pts'] for p in players
            ) -
            4 * (hits[w] - wildcard[w] - freehit[w])
    ) for w in gameweeks)

ft = so.expr_sum(
    (np.power(decay_gameweek, w - start - 1) if objective_type == 'linear' else 1) *
    (
        2 * rolling_transfers[w] # Value of having 2FT
    ) for w in gameweeks[1:]) # Value is added to the GW when a FT is rolled so exclude the first Gw 

if bboost_gw + 1:
    xp_bb = (np.power(decay_gameweek, bboost_gw) if objective_type == 'linear' else 1) * (
                so.expr_sum(
                    ((1 - decay_bench) * (team[p, start + bboost_gw] - starter[p, start + bboost_gw])) *
                    data.loc[p, str(start + bboost_gw) + '_Pts'] for p in players
                )
        )
else:
    xp_bb = 0

model.set_objective(- xp - ft - xp_bb, name='total_xp_obj', sense='N')

# Initial conditions: set team and FT depending on the team
model.add_constraints((team[p, start - 1] == 1 for p in initial_team), name='initial_team')
model.add_constraint(free_transfers[start - 1] == rolling_transfer + 1, name='initial_ft')


# Constraints
# The cost of the squad must exceed the budget
model.add_constraints((so.expr_sum(team[p, w] * data.loc[p, 'SV'] for p in players) <= budget for w in all_gameweeks), name='budget')

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

# The player must not be sold and bought simultaneously (on wildcard/freehit)
model.add_constraints((sell[p, w] + buy[p, w] <= 1 for p in players for w in gameweeks), name='single_buy_or_sell')


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


if freehit_gw + 1:
    # The chip must be used on the defined gameweek
    model.add_constraint(freehit[start + freehit_gw] == hits[start + freehit_gw], name='initial_freehit')
    model.add_constraint(freehit[start + freehit_gw + 1] == hits[start + freehit_gw], name='initial_freehit2')
    # The chip must only be used once
    model.add_constraint(so.expr_sum(freehit[w] for w in gameweeks) == hits[start + freehit_gw] + hits[start + freehit_gw + 1], name='freehit_once')
    # The freehit team must be kept only one gameweek
    model.add_constraints((buy[p, start + freehit_gw] == sell[p, start + freehit_gw + 1] for p in players), name='freehit1')
    model.add_constraints((sell[p, start + freehit_gw] == buy[p, start + freehit_gw + 1] for p in players), name='freehit2')
else:
    # The unused chip must not contribute
    model.add_constraint(so.expr_sum(freehit[w] for w in gameweeks) == 0, name='freehit_unused')

if wildcard_gw + 1:
    # The chip must be used on the defined gameweek
    model.add_constraint(wildcard[start + wildcard_gw] == hits[start + wildcard_gw], name='initial_wildcard')
    # The chip must only be used once
    model.add_constraint(so.expr_sum(wildcard[w] for w in gameweeks) == hits[start + wildcard_gw], name='wc_once')
else:
    # The unused chip must not contribute
    model.add_constraint(so.expr_sum(wildcard[w] for w in gameweeks) == 0, name='wildcard_unused')

if bboost_gw + 1:
    # The chip must be used on the defined gameweek
    model.add_constraint(bboost[start + bboost_gw] == 1, name='initial_bboost')
    # The chip must only be used once
    model.add_constraint(so.expr_sum(bboost[w] for w in gameweeks) == 1, name='bboost_once')
else:
    # The unused chip must not contribute
    model.add_constraint(so.expr_sum(bboost[w] for w in gameweeks) == 0, name='bboost_unused')
    
if threexc_gw + 1:
    # The chip must be used on the defined gameweek
    model.add_constraint(so.expr_sum(threexc[p, start + threexc_gw] for p in players) == 1, name='initial_3xc')
    # The chips must only be used once
    model.add_constraint(so.expr_sum(threexc[p, w] for p in players for w in gameweeks) == 1, name='tc_once')
    # The TC player must be the captain
    model.add_constraints((threexc[p, w] <= captain[p, w] for p in players for w in gameweeks), name='3xc_is_cap')
else:
    # The unused chip must not contribute
    model.add_constraint(so.expr_sum(threexc[p, w] for p in players for w in gameweeks) == 0, name='tc_unused')

for bias in love:
    if bias == 'buy' and love[bias]:
        assert all([w in gameweeks for (_, w) in love['buy']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in love['buy']]), 'Player selected to buy does not exist.'
        # The forced-buy player must be bought 
        model.add_constraints((buy[p, w] == 1 for (p, w) in love[bias]), name="force_buy")
    if bias == 'start'and love[bias]:
        print([w for w in love['start']])
        assert all([w in gameweeks for (_, w) in love['start']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in love['start']]), 'Player selected to start does not exist.'
        # The forced-in team player must be in the team 
        model.add_constraints((team[p, w] == 1 for (p, w) in love[bias]), name="force_in")
    if bias == 'team' and love[bias]:
        assert all([w in gameweeks for (_, w) in love['team']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in love['team']]), 'Player selected to be in the team does not exist.'
        # The forced-in starter player must be a starter 
        model.add_constraints((starter[p, w] == 1 for (p, w) in love[bias]), name="force_starter")
    if bias == 'cap' and love[bias]:
        assert all([w in gameweeks for (_, w) in love['cap']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in love['cap']]), 'Player selected to be the captain does not exist.'
        # The forced-in cap player must be the captain
        model.add_constraints((captain[p, w] == 1 for (p, w) in love[bias]), name="force_captain")

for bias in hate:
    if bias == 'sell' and hate[bias]:
        assert all([w in gameweeks for (_, w) in hate['sell']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in hate['sell']]), 'Player selected to sell does not exist.'
        # The forced-out player must be sold 
        model.add_constraints((sell[p, w] == 1 for (p, w) in hate[bias]), name="force_sell")
    if bias == 'bench' and hate[bias]:
        assert all([w in gameweeks for (_, w) in hate['bench']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in hate['bench']]), 'Player selected to start does not exist.'
        # The forced-out of starter player must not be starting
        model.add_constraints((starter[p, w] == 0 for (p, w) in hate[bias]), name="force_bench") # Force player out by a certain gw
    if bias == 'team' and hate[bias]:
        assert all([w in gameweeks for (_, w) in hate['team']]), 'Gameweek selected does not exist.'
        assert all([bias[0] in players for bias in hate['team']]), 'Player selected to be out of the team does not exist.'
        # The forced-out of team player must not be in team
        model.add_constraints((team[p, w] == 0 for (p, w) in hate[bias]), name="force_out") # Force player out by a certain gw (the player can get transfered sooner )

for bias in hit_limit:
    if bias == 'max' and hit_limit[bias]:
        assert all([w in gameweeks for (w, _) in hit_limit['max']]), 'Gameweek selected does not exist.'
        # The number of hits under the maximum
        model.add_constraints((hits[w] < max_hit for (w, max_hit) in hit_limit[bias]), name='hits_max')
    if bias == 'eq' and hit_limit[bias]:
        assert all([w in gameweeks for (w, _) in hit_limit['eq']]), 'Gameweek selected does not exist.'
        # The number of hits equal to the choice
        model.add_constraints((hits[w] == nb_hit for (w, nb_hit) in hit_limit[bias]), name='hits_eq')
    if bias == 'min' and hit_limit[bias]:
        assert all([w in gameweeks for (w, _) in hit_limit['min']]), 'Gameweek selected does not exist.'
        # The number of hits above the minumum
        model.add_constraints((hits[w] > min_hit for (w, min_hit) in hit_limit[bias]), name='hits_min')

for gw in two_ft_gw:
    assert gw > start and gw <= start + horizon, 'Gameweek selected cannot be constrained.'
    # Force rolling free transfer
    model.add_constraint(free_transfers[gw] == 2, name=f'force_roll_{gw}')

target = 'Top_250K'
if not nb_differentials:
    print("*****")
    data['Differential'] = np.where(data[target] < differential_threshold, 1, 0)
    model.add_constraints(
        (
            so.expr_sum(starter[p, w] * data.loc[p, 'Differential'] for p in players) >= nb_differentials for w in gameweeks
        ), name='differentials')


for i in range(nb_suboptimal):
    # Solve
    model.export_mps(filename=f"optimization/tmp/{model_name}.mps")
    command = f'cbc optimization/tmp/{model_name}.mps solve solu optimization/tmp/{model_name}_solution.txt'
    # command = f'cbc optimization/tmp/{model_name}.mps cost column solve solu optimization/tmp/{model_name}_solution.txt'

    if log:
        os.system(command)
    else:
        process = Popen(command, shell=True, stdout=DEVNULL)
        process.wait()

    # Reset variables for next passes
    for v in model.get_variables():
        v.set_value(0)

    with open(f'optimization/tmp/{model_name}_solution.txt', 'r') as f:
        for line in f:
            if 'objective value' in line:
                continue
            words = line.split()
            var = model.get_variable(words[1])
            var.set_value(float(words[2]))

    # GW
    print(f"\n----- {i} -----")
    pretty_print(data, start, period, team, starter, captain, vicecaptain, buy, sell, free_transfers,
                hits, freehit_gw, wildcard_gw, bboost_gw, threexc_gw, i)

    if i != nb_suboptimal - 1:
        # Select the players that have been transfered in/out
        if cutoff_search == 'first_buy':
            actions = so.expr_sum(buy[p, start] for p in players if buy[p, start].get_value() > 0.5)
            gw_range = [start]
        elif cutoff_search == 'horizon_buy':
            actions = so.expr_sum(so.expr_sum(buy[p, w] for p in players if buy[p, w].get_value() > 0.5) for w in gameweeks)
            gw_range = gameweeks
        elif cutoff_search == 'first_transfer':
            actions = (
                so.expr_sum(buy[p, start] for p in players if buy[p, start].get_value() > 0.5) +
                so.expr_sum(sell[p, start] for p in players if sell[p, start].get_value() > 0.5)
                )
            gw_range = [start]
        elif cutoff_search == 'horizon_transfer':
            actions = (
                so.expr_sum(so.expr_sum(buy[p, w] for p in players if buy[p, w].get_value() > 0.5) for w in gameweeks) +
                so.expr_sum(so.expr_sum(sell[p, w] for p in players if sell[p, w].get_value() > 0.5) for w in gameweeks)
                )
            gw_range = gameweeks

        if actions.get_value() != 0:
            # This step forces one transfer to be unfeasible
            # Note: the constraint is only applied to the activated transfers
            # so the ones not activated are thus allowed.
            model.add_constraint(actions <= actions.get_value() - 1, name=f'cutoff_{i}')
        else:
            # Force one transfer in case of sub-optimal solution choosing to roll transfer
            model.add_constraint(so.expr_sum(number_of_transfers[w] for w in gw_range) >= 1, name=f'cutoff_{i}')
