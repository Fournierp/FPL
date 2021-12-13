import pandas as pd
import numpy as np

import os
from subprocess import Popen, DEVNULL
import sasoptpy as so
import logging

from utils import (
    get_team,
    get_predictions,
    get_rolling,
    pretty_print,
    get_chips,
    get_next_gw,
    get_ownership_data,
    randomize)


class Team_Planner:

    def __init__(self, team_id=35868, horizon=5, noise=False):
        self.horizon = horizon
        self.get_data(team_id)

        if noise:
            self.random_noise(None)

    def get_data(self, team_id):
        # Data collection
        # Predicted points from https://fplreview.com/
        df = get_predictions()
        self.team_names = df.columns[-20:].values
        self.data = df.copy().set_index('id')
        
        # Ownership data
        ownership = get_ownership_data()
        self.data = pd.concat([self.data, ownership], axis=1, join="inner")
        self.players = self.data.index.tolist()

        # FPL data
        self.start = get_next_gw()
        self.initial_team, self.bank = get_team(team_id, self.start - 1)
        self.freehit_used, self.wildcard_used, self.bboost_used, self.threexc_used = get_chips(team_id, self.start - 1)

        # GW
        self.period = min(self.horizon, len([col for col in df.columns if '_Pts' in col]))
        self.rolling_transfer, self.transfer = get_rolling(team_id, self.start - 1) 
        self.budget = np.sum([self.data.loc[p, 'SV'] for p in self.initial_team]) + self.bank
        self.all_gameweeks = np.arange(self.start-1, self.start+self.period)
        self.gameweeks = np.arange(self.start, self.start+self.period)

        # Sort DF by EV for efficient optimization
        self.data['total_ev'] = self.data[[col for col in df.columns if '_Pts' in col]].sum(axis=1)
        self.data.sort_values(by=['total_ev'], ascending=[False], inplace=True)

    def random_noise(self, seed):
        # Apply random noise
        self.data = randomize(seed, self.data, self.start)

    def build_model(self, model_name, objective_type='decay', decay_gameweek=0.9, vicecap_decay=0.1, decay_bench=[0.1, 0.1, 0.1, 0.1], ft_val=0):
        # Model
        self.model = so.Model(name=f'{model_name}_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(self.players, self.all_gameweeks, name='team', vartype=so.binary)
        self.starter = self.model.add_variables(self.players, self.gameweeks, name='starter', vartype=so.binary)
        self.bench = self.model.add_variables(self.players, self.gameweeks, order, name='bench', vartype=so.binary)
        self.captain = self.model.add_variables(self.players, self.gameweeks, name='captain', vartype=so.binary)
        self.vicecaptain = self.model.add_variables(self.players, self.gameweeks, name='vicecaptain', vartype=so.binary)

        self.buy = self.model.add_variables(self.players, self.gameweeks, name='buy', vartype=so.binary)
        self.sell = self.model.add_variables(self.players, self.gameweeks, name='sell', vartype=so.binary)

        self.free_transfers = self.model.add_variables(self.all_gameweeks, name='ft', vartype=so.integer, lb=1, ub=2)
        self.hits = self.model.add_variables(self.gameweeks, name='hits', vartype=so.integer, lb=0, ub=15)
        self.rolling_transfers = self.model.add_variables(self.gameweeks, name='rolling', vartype=so.binary)

        # Objective: maximize total expected points
        # Assume a % (decay_bench) chance of a player being subbed on
        # Assume a % (decay_gameweek) reliability of next week's xPts
        xp = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) *
            (
                    so.expr_sum(
                        (
                            self.starter[p, w] + self.captain[p, w] +
                            (vicecap_decay * self.vicecaptain[p, w]) +
                            so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                        ) *
                        self.data.loc[p, f'{w}_Pts'] for p in self.players
                    ) -
                    4 * self.hits[w]
            ) for w in self.gameweeks)

        ftv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1) *
            (
                ft_val * self.rolling_transfers[w] # Value of having 2FT
            ) for w in self.gameweeks[1:]) # Value is added to the GW when a FT is rolled so exclude the first Gw 

        self.model.set_objective(- xp - ftv, name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints((self.team[p, self.start - 1] == 1 for p in self.initial_team), name='initial_team')
        self.model.add_constraint(self.free_transfers[self.start - 1] == self.rolling_transfer + 1, name='initial_ft')

        # Constraints
        # The cost of the squad must exceed the budget
        self.model.add_constraints((so.expr_sum(self.team[p, w] * self.data.loc[p, 'SV'] for p in self.players) <= self.budget for w in self.all_gameweeks), name='budget')

        # The number of players must be 11 on field, 4 on bench, 1 captain & 1 vicecaptain
        self.model.add_constraints((so.expr_sum(self.team[p, w] for p in self.players) == 15 for w in self.all_gameweeks), name='15_players')
        self.model.add_constraints((so.expr_sum(self.starter[p, w] for p in self.players) == 11 for w in self.gameweeks), name='11_starters')
        self.model.add_constraints((so.expr_sum(self.captain[p, w] for p in self.players) == 1 for w in self.gameweeks), name='1_captain')
        self.model.add_constraints((so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1 for w in self.gameweeks), name='1_vicecaptain')

        # Bench constraints
        self.model.add_constraints((so.expr_sum(self.bench[p, w, 0] for p in self.players if self.data.loc[p, 'G'] == 1) == 1 for w in self.gameweeks), name='one_bench_gk')
        self.model.add_constraints((so.expr_sum(self.bench[p, w, o] for p in self.players) == 1 for w in self.gameweeks for o in [1, 2, 3]), name='one_per_bench_spot')
        self.model.add_constraints((self.bench[p, w, o] <= self.team[p, w] for p in self.players for w in self.gameweeks for o in order), name='bench_in_team')
        self.model.add_constraints((self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order) <= 1 for p in self.players for w in self.gameweeks), name='bench_not_a_starter')

        # A captain must not be picked more than once
        self.model.add_constraints((self.captain[p, w] + self.vicecaptain[p, w] <= 1 for p in self.players for w in self.gameweeks), name='cap_or_vice')

        # The number of players from a team must not be more than three
        self.model.add_constraints((so.expr_sum(self.team[p, w] * self.data.loc[p, team_name] for p in self.players) <= 3
                            for team_name in self.team_names for w in self.gameweeks), name='team_limit')

        # The number of players fit the requirements 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints((so.expr_sum(self.team[p, w] * self.data.loc[p, 'G'] for p in self.players) == 2 for w in self.gameweeks), name='gk_limit')
        self.model.add_constraints((so.expr_sum(self.team[p, w] * self.data.loc[p, 'D'] for p in self.players) == 5 for w in self.gameweeks), name='def_limit')
        self.model.add_constraints((so.expr_sum(self.team[p, w] * self.data.loc[p, 'M'] for p in self.players) == 5 for w in self.gameweeks), name='mid_limit')
        self.model.add_constraints((so.expr_sum(self.team[p, w] * self.data.loc[p, 'F'] for p in self.players) == 3 for w in self.gameweeks), name='for_limit')

        # The formation is valid i.e. Minimum one goalkeeper, 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints((so.expr_sum(self.starter[p, w] * self.data.loc[p, 'G'] for p in self.players) == 1 for w in self.gameweeks), name='gk_min')
        self.model.add_constraints((so.expr_sum(self.starter[p, w] * self.data.loc[p, 'D'] for p in self.players) >= 3 for w in self.gameweeks), name='def_min')
        self.model.add_constraints((so.expr_sum(self.starter[p, w] * self.data.loc[p, 'M'] for p in self.players) >= 2 for w in self.gameweeks), name='mid_min')
        self.model.add_constraints((so.expr_sum(self.starter[p, w] * self.data.loc[p, 'F'] for p in self.players) >= 1 for w in self.gameweeks), name='for_min')

        # The captain & vicecap must be a player on the field
        self.model.add_constraints((self.captain[p, w] <= self.starter[p, w] for p in self.players for w in self.gameweeks), name='captain_in_starters')
        self.model.add_constraints((self.vicecaptain[p, w] <= self.starter[p, w] for p in self.players for w in self.gameweeks), name='vicecaptain_in_starters')

        # The starters must be in the team
        self.model.add_constraints((self.starter[p, w] <= self.team[p, w] for p in self.players for w in self.gameweeks), name='starters_in_team')

        # The team must be equal to the next week excluding transfers
        self.model.add_constraints((self.team[p, w] == self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] for p in self.players for w in self.gameweeks),
                            name='team_transfer')

        # The rolling transfer must be equal to the number of free transfers not used (+ 1)
        self.model.add_constraints((self.free_transfers[w] == self.rolling_transfers[w] + 1 for w in self.gameweeks), name='rolling_ft_rel')

        # The player must not be sold and bought simultaneously (on wildcard/freehit)
        self.model.add_constraints((self.sell[p, w] + self.buy[p, w] <= 1 for p in self.players for w in self.gameweeks), name='single_buy_or_sell')

        # Rolling transfers
        self.number_of_transfers = {w: so.expr_sum(self.sell[p, w] for p in self.players) for w in self.gameweeks}
        self.number_of_transfers[self.start - 1] = self.transfer
        self.model.add_constraints((self.free_transfers[w - 1] - self.number_of_transfers[w - 1] <= 2 * self.rolling_transfers[w] for w in self.gameweeks),
                            name='rolling_condition_1')
        self.model.add_constraints(
            (self.free_transfers[w - 1] - self.number_of_transfers[w - 1] >= self.rolling_transfers[w] + (-14) * (1 - self.rolling_transfers[w])
            for w in self.gameweeks),
            name='rolling_condition_2')

        # The number of hits must be the number of transfer except the free ones.
        self.model.add_constraints((self.hits[w] >= self.number_of_transfers[w] - self.free_transfers[w] for w in self.gameweeks), name='hits')

    def differential_model(self, nb_differentials=3, threshold=10, target='Top_100K'):
        self.data['Differential'] = np.where(self.data[target] < threshold, 1, 0)
        # A min numberof starter players must be differentials
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'Differential'] for p in self.players) >= nb_differentials for w in self.gameweeks
            ), name='differentials')

    def select_chips_model(self, freehit_gw, wildcard_gw, bboost_gw, threexc_gw, objective_type='decay', decay_gameweek=0.9, vicecap_decay=0.1, decay_bench=[0.1, 0.1, 0.1, 0.1]):
        assert (freehit_gw < self.horizon), "Select a gameweek within the horizon."
        assert (wildcard_gw < self.horizon), "Select a gameweek within the horizon."
        assert (bboost_gw < self.horizon), "Select a gameweek within the horizon."
        assert (threexc_gw < self.horizon), "Select a gameweek within the horizon."

        assert not (self.freehit_used and freehit_gw >= 0), "Freehit chip was already used."
        assert not (self.wildcard_used and wildcard_gw >= 0), "Wildcard chip was already used."
        assert not (self.bboost_used and bboost_gw >= 0), "Bench boost chip was already used."
        assert not (self.threexc_used and threexc_gw >= 0), "Tripple captain chip was already used."
    
        freehit = self.model.add_variables(self.gameweeks, name='fh', vartype=so.integer, lb=0, ub=15)
        wildcard = self.model.add_variables(self.gameweeks, name='wc', vartype=so.integer, lb=0, ub=15)
        bboost = self.model.add_variables(self.gameweeks, name='bb', vartype=so.binary)
        threexc = self.model.add_variables(self.players, self.gameweeks, name='3xc', vartype=so.binary)

        order = [0, 1, 2, 3]

        # Objective: maximize total expected points
        # Assume a 10% (decay_bench) chance of a player not playing
        # Assume a 80% (decay_gameweek) reliability of next week's xPts
        xp = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) *
            (
                    so.expr_sum(
                        (
                            self.starter[p, w] + self.captain[p, w] + threexc[p, w] +
                            (vicecap_decay * self.vicecaptain[p, w]) +
                            so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                        ) *
                        self.data.loc[p, f'{w}_Pts'] for p in self.players
                    ) -
                    4 * (self.hits[w] - wildcard[w] - freehit[w])
            ) for w in self.gameweeks)

        if bboost_gw + 1:
            xp_bb = (np.power(decay_gameweek, bboost_gw) if objective_type == 'linear' else 1) * (
                        so.expr_sum(
                            so.expr_sum((1 - decay_bench[o]) * self.bench[p, self.start + bboost_gw, o] for o in order) *
                            self.data.loc[p, f'{self.start + bboost_gw}_Pts'] for p in self.players
                        )
                )
        else:
            xp_bb = 0

        self.model.set_objective(- xp - xp_bb, name='total_xp_obj', sense='N')

        if freehit_gw + 1:
            # The chip must be used on the defined gameweek
            self.model.add_constraint(freehit[self.start + freehit_gw] == self.hits[self.start + freehit_gw], name='initial_freehit')
            self.model.add_constraint(freehit[self.start + freehit_gw + 1] == self.hits[self.start + freehit_gw], name='initial_freehit2')
            # The chip must only be used once
            self.model.add_constraint(so.expr_sum(freehit[w] for w in self.gameweeks) == self.hits[self.start + freehit_gw] + self.hits[self.start + freehit_gw + 1], name='freehit_once')
            # The freehit team must be kept only one gameweek
            self.model.add_constraints((self.buy[p, self.start + freehit_gw] == self.sell[p, self.start + freehit_gw + 1] for p in self.players), name='freehit1')
            self.model.add_constraints((self.sell[p, self.start + freehit_gw] == self.buy[p, self.start + freehit_gw + 1] for p in self.players), name='freehit2')
        else:
            # The unused chip must not contribute
            self.model.add_constraint(so.expr_sum(freehit[w] for w in self.gameweeks) == 0, name='freehit_unused')

        if wildcard_gw + 1:
            # The chip must be used on the defined gameweek
            self.model.add_constraint(wildcard[self.start + wildcard_gw] == self.hits[self.start + wildcard_gw], name='initial_wildcard')
            # The chip must only be used once
            self.model.add_constraint(so.expr_sum(wildcard[w] for w in self.gameweeks) == self.hits[self.start + wildcard_gw], name='wc_once')
        else:
            # The unused chip must not contribute
            self.model.add_constraint(so.expr_sum(wildcard[w] for w in self.gameweeks) == 0, name='wildcard_unused')

        if bboost_gw + 1:
            # The chip must be used on the defined gameweek
            self.model.add_constraint(bboost[self.start + bboost_gw] == 1, name='initial_bboost')
            # The chip must only be used once
            self.model.add_constraint(so.expr_sum(bboost[w] for w in self.gameweeks) == 1, name='bboost_once')
        else:
            # The unused chip must not contribute
            self.model.add_constraint(so.expr_sum(bboost[w] for w in self.gameweeks) == 0, name='bboost_unused')
            
        if threexc_gw + 1:
            # The chip must be used on the defined gameweek
            self.model.add_constraint(so.expr_sum(threexc[p, self.start + threexc_gw] for p in self.players) == 1, name='initial_3xc')
            # The chips must only be used once
            self.model.add_constraint(so.expr_sum(threexc[p, w] for p in self.players for w in self.gameweeks) == 1, name='tc_once')
            # The TC player must be the captain
            self.model.add_constraints((threexc[p, w] <= self.captain[p, w] for p in self.players for w in self.gameweeks), name='3xc_is_cap')
        else:
            # The unused chip must not contribute
            self.model.add_constraint(so.expr_sum(threexc[p, w] for p in self.players for w in self.gameweeks) == 0, name='tc_unused')

    def biased_model(self, love, hate, hit_limit, two_ft_gw):
        force_in = self.model.add_variables(self.players, self.gameweeks, name='fi', vartype=so.binary)
        force_out = self.model.add_variables(self.players, self.gameweeks, name='fo', vartype=so.binary)

        for bias in love:
            if bias == 'buy' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['buy']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['buy']]), 'Player selected to buy does not exist.'
                # The forced-buy player must be bought 
                self.model.add_constraints((buy[p, w] == 1 for (p, w) in love[bias]), name="force_buy")
            if bias == 'start'and love[bias]:
                print([w for w in love['start']])
                assert all([w in self.gameweeks for (_, w) in love['start']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['start']]), 'Player selected to start does not exist.'
                # The forced-in team player must be in the team 
                self.model.add_constraints((team[p, w] == 1 for (p, w) in love[bias]), name="force_in")
            if bias == 'team' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['team']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['team']]), 'Player selected to be in the team does not exist.'
                # The forced-in starter player must be a starter 
                self.model.add_constraints((self.starter[p, w] == 1 for (p, w) in love[bias]), name="force_starter")
            if bias == 'cap' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['cap']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['cap']]), 'Player selected to be the captain does not exist.'
                # The forced-in cap player must be the captain
                self.model.add_constraints((self.captain[p, w] == 1 for (p, w) in love[bias]), name="force_captain")

        for bias in hate:
            if bias == 'sell' and hate[bias]:
                assert all([w in self.gameweeks for (_, w) in hate['sell']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in hate['sell']]), 'Player selected to sell does not exist.'
                # The forced-out player must be sold 
                self.model.add_constraints((sell[p, w] == 1 for (p, w) in hate[bias]), name="force_sell")
            if bias == 'bench' and hate[bias]:
                assert all([w in self.gameweeks for (_, w) in hate['bench']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in hate['bench']]), 'Player selected to start does not exist.'
                # The forced-out of starter player must not be starting
                self.model.add_constraints((self.starter[p, w] == 0 for (p, w) in hate[bias]), name="force_bench") # Force player out by a certain gw
            if bias == 'team' and hate[bias]:
                assert all([w in self.gameweeks for (_, w) in hate['team']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in hate['team']]), 'Player selected to be out of the team does not exist.'
                # The forced-out of team player must not be in team
                self.model.add_constraints((self.team[p, w] == 0 for (p, w) in hate[bias]), name="force_out") # Force player out by a certain gw (the player can get transfered sooner )

        for bias in hit_limit:
            if bias == 'max' and hit_limit[bias]:
                assert all([w in self.gameweeks for (w, _) in hit_limit['max']]), 'Gameweek selected does not exist.'
                # The number of hits under the maximum
                self.model.add_constraints((self.hits[w] < max_hit for (w, max_hit) in hit_limit[bias]), name='hits_max')
            if bias == 'eq' and hit_limit[bias]:
                assert all([w in self.gameweeks for (w, _) in hit_limit['eq']]), 'Gameweek selected does not exist.'
                # The number of hits equal to the choice
                self.model.add_constraints((self.hits[w] == nb_hit for (w, nb_hit) in hit_limit[bias]), name='hits_eq')
            if bias == 'min' and hit_limit[bias]:
                assert all([w in self.gameweeks for (w, _) in hit_limit['min']]), 'Gameweek selected does not exist.'
                # The number of hits above the minumum
                self.model.add_constraints((self.hits[w] > min_hit for (w, min_hit) in hit_limit[bias]), name='hits_min')

        for gw in two_ft_gw:
            assert gw > self.start and gw <= self.start + self.horizon, 'Gameweek selected cannot be constrained.'
            # Force rolling free transfer
            self.model.add_constraint(self.free_transfers[gw] == 2, name=f'force_roll_{gw}')

    def solve(self, model_name, log=False):
        self.model.export_mps(filename=f"optimization/tmp/{model_name}.mps")
        command = f'cbc optimization/tmp/{model_name}.mps solve solu optimization/tmp/{model_name}_solution.txt'

        if log:
            os.system(command)
        else:
            process = Popen(command, shell=True, stdout=DEVNULL)
            process.wait()

        # Reset variables for next passes
        for v in self.model.get_variables():
            v.set_value(0)

        with open(f'optimization/tmp/{model_name}_solution.txt', 'r') as f:
            for line in f:
                if 'objective value' in line:
                    continue
                words = line.split()
                var = self.model.get_variable(words[1])
                var.set_value(float(words[2]))

        pretty_print(
            self.data, self.start, self.period,
            self.team, self.starter,
            self.bench,
            self.captain, self.vicecaptain,
            self.buy, self.sell,
            self.free_transfers, self.hits,
            freehit=-1, wildcard=-1,
            bboost=-1, threexc=-1,
            nb_suboptimal=0)

    def suboptimals(self, model_name, iterations=3, cutoff_search='first_transfer'):
        for i in range(iterations):

            print(f"\n----- {i} -----")
            self.solve(model_name + f'_{i}')

            if i != iterations - 1:
                # Select the players that have been transfered in/out
                if cutoff_search == 'first_buy':
                    actions = so.expr_sum(self.buy[p, self.start] for p in self.players if self.buy[p, self.start].get_value() > 0.5)
                    gw_range = [self.start]
                elif cutoff_search == 'horizon_buy':
                    actions = so.expr_sum(so.expr_sum(self.buy[p, w] for p in self.players if self.buy[p, w].get_value() > 0.5) for w in self.gameweeks)
                    gw_range = self.gameweeks
                elif cutoff_search == 'first_transfer':
                    actions = (
                        so.expr_sum(self.buy[p, self.start] for p in self.players if self.buy[p, self.start].get_value() > 0.5) +
                        so.expr_sum(self.sell[p, self.start] for p in self.players if self.sell[p, self.start].get_value() > 0.5)
                        )
                    gw_range = [self.start]
                elif cutoff_search == 'horizon_transfer':
                    actions = (
                        so.expr_sum(so.expr_sum(self.buy[p, w] for p in self.players if self.buy[p, w].get_value() > 0.5) for w in self.gameweeks) +
                        so.expr_sum(so.expr_sum(self.sell[p, w] for p in self.players if self.sell[p, w].get_value() > 0.5) for w in self.gameweeks)
                        )
                    gw_range = self.gameweeks

                if actions.get_value() != 0:
                    # This step forces one transfer to be unfeasible
                    # Note: the constraint is only applied to the activated transfers
                    # so the ones not activated are thus allowed.
                    self.model.add_constraint(actions <= actions.get_value() - 1, name=f'cutoff_{i}')
                else:
                    # Force one transfer in case of sub-optimal solution choosing to roll transfer
                    self.model.add_constraint(so.expr_sum(self.number_of_transfers[w] for w in gw_range) >= 1, name=f'cutoff_{i}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    tp = Team_Planner(
        team_id=35868,
        horizon=5,
        noise=False)

    tp.build_model(
        model_name="vanilla",
        objective_type='decay',
        decay_gameweek=0.85,
        vicecap_decay=0.1,
        decay_bench=[0.03, 0.21, 0.06, 0.002],
        ft_val=1.5)

    # tp.differential_model(
    #     nb_differentials=3,
    #     threshold=10,
    #     target='Top_100K')

    # Chip strategy: set to (-1) if you don't want to use
    # Choose a value in range [0-horizon] as the number of gameweeks after the current one
    # tp.select_chips_model(
    #     freehit_gw=-1,
    #     wildcard_gw=-1,
    #     bboost_gw=-1,
    #     threexc_gw=-1,
    #     objective_type='decay',
    #     decay_gameweek=0.85,
    #     vicecap_decay=0.1,
    #     decay_bench=[0.03, 0.21, 0.06, 0.002])

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
    # Example usage: {(gw, amount)}: {(17, 5)}
    hit_limit = {
        'max': {},
        'eq': {},
        'min': {}
    }

    # Gameweeks where 2FT are desired. In other words the (GW-1) where a FT is rolled.
    two_ft_gw = []

    # tp.biased_model(
    #     love=love,
    #     hate=hate,
    #     hit_limit=hit_limit,
    #     two_ft_gw=two_ft_gw)

    # tp.solve(
    #     model_name="vanilla",
    #     log=False)

    tp.suboptimals(
        model_name="vanilla",
        iterations=3,
        cutoff_search='first_transfer')
