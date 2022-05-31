import pandas as pd
import numpy as np

import os
from subprocess import Popen, DEVNULL
import sasoptpy as so
import logging
import json
from concurrent.futures import ProcessPoolExecutor

from utils import (
    get_team,
    get_predictions,
    get_rolling,
    pretty_print,
    get_chips,
    get_next_gw,
    get_ownership_data,
    randomize)


class Team_Optimization:
    """ Mathematical optimization of FPL """

    def __init__(self, team_id=35868, horizon=5, noise=False, premium=False):
        """

        Args:
            team_id (int): Team to optimize
            horizon (int): Planning horizon
            noise (bool): Apply noise
            premium (bool, optional): Load premium data.
        """
        self.horizon = horizon
        self.premium = premium
        self.get_data(team_id)

        if noise:
            self.random_noise(None)

    def get_data(self, team_id):
        """ Get EV& Ownership data along with team data

        Args:
            team_id (int): Team to optimize
        """
        # Data collection
        # Predicted points from https://fplreview.com/
        df = get_predictions(premium=self.premium)
        self.team_names = df.columns[-20:].values
        self.data = df.copy()

        # Ownership data
        ownership = get_ownership_data()
        self.data = pd.concat([self.data, ownership], axis=1, join="inner")

        # FPL data
        self.start = get_next_gw()
        self.initial_team, self.bank = get_team(team_id, self.start-1)
        (
            self.freehit_used,
            self.wildcard_used,
            self.bboost_used,
            self.threexc_used
            ) = get_chips(team_id, self.start - 1)

        # GW
        self.period = min(
            self.horizon,
            len([col for col in df.columns if '_Pts' in col]))
        (
            self.rolling_transfer,
            self.transfer
            ) = get_rolling(team_id, self.start - 1)
        self.budget = np.sum(
            [self.data.loc[p, 'SV'] for p in self.initial_team]
            ) + self.bank
        self.all_gameweeks = np.arange(self.start-1, self.start+self.period)
        self.gameweeks = np.arange(self.start, self.start+self.period)

        # Sort DF by EV for efficient optimization
        self.data['total_ev'] = self.data[
            [col for col in df.columns if '_Pts' in col]
            ].sum(axis=1)
        self.data.sort_values(by=['total_ev'], ascending=[False], inplace=True)

        # Drop players that are not predicted to play much to reduce the search space
        self.data.drop(self.data[self.data.total_ev <= 1].index, inplace=True)
        self.players = self.data.index.tolist()

        self.initial_team_df = pd.DataFrame(
            [],
            columns=['GW', 'Name', 'Pos', 'Team', 'SV'])

        for p in self.initial_team:
            self.initial_team_df = self.initial_team_df.append(
                {
                    'GW': self.start-1,
                    'Name': self.data.loc[p]['Name'],
                    'Pos': self.data.loc[p]['Pos'],
                    'Team': self.data.loc[p]['Team'],
                    'SV': self.data.loc[p]['SV']},
                ignore_index=True)

    def random_noise(self, seed):
        """ Apply random Normal noise to EV Data

        Args:
            seed (int): Seed the RNG
        """
        # Apply random noise
        self.data = randomize(seed, self.data, self.start)

    def build_model(
            self,
            model_name,
            freehit_gw=-1,
            wildcard_gw=-1,
            bboost_gw=-1,
            threexc_gw=-1,
            objective_type='decay',
            decay_gameweek=0.9,
            vicecap_decay=0.1,
            decay_bench=[0.1, 0.1, 0.1, 0.1],
            ft_val=0,
            itb_val=0,
            hit_val=6):
        """ Build the core of linear optimization model

        Args:
            model_name (string): Model name
            freehit_gw (int): Gw to use chip in
            wildcard_gw (int): Gw to use chip in
            bboost_gw (int): Gw to use chip in
            threexc_gw (int): Gw to use chip in
            objective_type (str): Decay to apply higher importance to early GW
                and Linear to apply uniform weights
            decay_gameweek (float): Weight decay per gameweek
            vicecap_decay (float): Weight applied to points scored by vice
            decay_bench (list): Weight applied to points scored by bench.
            ft_val (int): Value of rolling a transfer.
            itb_val (int): Value of having money in the bank.
            hit_val (int): Penalty of taking a hit.
        """
        assert (freehit_gw < self.horizon), "Select a GW within the horizon."
        assert (wildcard_gw < self.horizon), "Select a GW within the horizon."
        assert (bboost_gw < self.horizon), "Select a GW within the horizon."
        assert (threexc_gw < self.horizon), "Select a GW within the horizon."

        assert not (self.freehit_used and freehit_gw >= 0), "Freehit chip was already used."
        assert not (self.wildcard_used and wildcard_gw >= 0), "Wildcard chip was already used."
        assert not (self.bboost_used and bboost_gw >= 0), "Bench boost chip was already used."
        assert not (self.threexc_used and threexc_gw >= 0), "Tripple captain chip was already used."

        # Model
        self.model = so.Model(name=model_name)

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='team',
            vartype=so.binary)
        self.team_fh = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='team_fh',
            vartype=so.binary)
        self.starter = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='starter',
            vartype=so.binary)
        self.bench = self.model.add_variables(
            self.players,
            self.gameweeks,
            order,
            name='bench',
            vartype=so.binary)

        self.captain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='captain',
            vartype=so.binary)
        self.vicecaptain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='vicecaptain',
            vartype=so.binary)

        self.buy = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='buy',
            vartype=so.binary)
        self.sell = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='sell',
            vartype=so.binary)

        self.triple = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='3xc',
            vartype=so.binary)
        self.bboost = self.model.add_variables(
            self.gameweeks,
            name='bb',
            vartype=so.binary)
        self.freehit = self.model.add_variables(
            self.gameweeks,
            name='fh',
            vartype=so.binary)
        self.wildcard = self.model.add_variables(
            self.gameweeks,
            name='wc',
            vartype=so.binary)

        self.aux = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='aux',
            vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start-1, self.start+self.period+1),
            name='ft',
            vartype=so.integer,
            lb=0)
        self.hits = self.model.add_variables(
            self.all_gameweeks,
            name='hits',
            vartype=so.integer,
            lb=0)
        self.in_the_bank = self.model.add_variables(
            self.all_gameweeks,
            name='itb',
            vartype=so.continuous,
            lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.starter[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        cap = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.captain[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        vice = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                vicecap_decay * self.vicecaptain[p, w] *
                self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        bench = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                * self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        txc = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                2 * self.triple[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        hits = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (hit_val * self.hits[w]) for w in self.gameweeks)

        ftv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                ft_val * (self.free_transfers[w] - 1)
            ) for w in self.gameweeks[1:])

        itbv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                itb_val * self.in_the_bank[w]
            ) for w in self.gameweeks)

        self.model.set_objective(
            - starter - cap - vice - bench - txc - ftv - itbv + hits,
            name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints(
            (self.team[p, self.start - 1] == 1 for p in self.initial_team),
            name='initial_team')
        self.model.add_constraint(
            self.free_transfers[self.start] == self.rolling_transfer + 1,
            name='initial_ft')
        self.model.add_constraint(
            self.in_the_bank[self.start - 1] == self.bank,
            name='initial_itb')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <=
            -- (not self.threexc_used),
            name='tc_once')
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <=
            -- (not self.bboost_used),
            name='bb_once')
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <=
            -- (not self.freehit_used),
            name='fh_once')
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <=
            -- (not self.wildcard_used),
            name='wc_once')

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players) +
                self.bboost[w] + self.freehit[w] + self.wildcard[w] <= 1
                for w in self.gameweeks),
            name='chip_once')

        # The chips must be used on the selected GW
        if bboost_gw + 1:
            self.model.add_constraint(
                self.bboost[self.start + bboost_gw] == 1,
                name='bboost_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0,
                name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, self.start + threexc_gw] for p in self.players
                    ) == 1,
                name='triple_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, w] for p in self.players for w in self.gameweeks
                    ) == 0,
                name='triple_unused')

        if freehit_gw + 1:
            self.model.add_constraint(
                self.freehit[self.start + freehit_gw] == 1,
                name='freehit_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0,
                name='freehit_unused')

        if wildcard_gw + 1:
            self.model.add_constraint(
                self.wildcard[self.start + wildcard_gw] == 1,
                name='wildcard_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.wildcard[w] for w in self.gameweeks) == 0,
                name='wildcard_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 for w in self.gameweeks),
            name='gk_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='def_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='mid_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 for w in self.gameweeks),
            name='for_limit')

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit')

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 * self.freehit[w]
                for w in self.gameweeks),
            name='gk_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='def_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='mid_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 * self.freehit[w]
                for w in self.gameweeks),
            name='for_limit_fh')

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3 * self.freehit[w]
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit_fh')

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] for p in self.players) == 11 +
                4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'G']
                    for p in self.players) ==
                1 + self.bboost[w] for w in self.gameweeks),
            name='gk_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'D']
                    for p in self.players) >= 3 for w in self.gameweeks),
            name='def_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'M']
                    for p in self.players) >= 2 for w in self.gameweeks),
            name='mid_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'F']
                    for p in self.players) >= 1 for w in self.gameweeks),
            name='for_min')

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (
                self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.24')
        self.model.add_constraints(
            (
                self.aux[p, w] <= self.team[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.25')
        self.model.add_constraints(
            (
                self.aux[p, w] <= 1 - self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='4.26')

        # Captain
        # One captain (or one triple cap) must be picked once
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.captain[p, w] + self.triple[p, w]
                    for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain')
        # One vice captain must be picked once
        self.model.add_constraints(
            (
                so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1
                for w in self.gameweeks),
            name='one_vicecaptain')
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] +
                self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players for w in self.gameweeks),
            name='cap_in_starters')

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.bench[p, w, 0] for p in self.players
                    if self.data.loc[p, 'G'] == 1) ==
                1 - self.bboost[w] for w in self.gameweeks),
            name='one_bench_gk')
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1
                for w in self.gameweeks for o in [0, 1, 2, 3]),
            name='one_per_bench_spot')

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='bench_team')
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players for w in self.gameweeks),
            name='bench_team_fh')

        # Budget
        sold_amount = {
            w: so.expr_sum(
                self.sell[p, w] * self.data.loc[p, 'SV'] for p in self.players)
            for w in self.gameweeks}
        bought_amount = {
            w: so.expr_sum(
                self.buy[p, w] * self.data.loc[p, 'BV'] for p in self.players)
            for w in self.gameweeks}
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] +
                sold_amount[w] - bought_amount[w] for w in self.gameweeks),
            name='budget')
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] ==
                self.team[p, w] for p in self.players for w in self.gameweeks),
            name='team_similarity')
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (
                self.sell[p, w] + self.buy[p, w] <= 1
                for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell')

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1] + so.expr_sum(
                    self.team[p, w - 1] * self.data.loc[p, 'SV']
                    for p in self.players) >=
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'BV']
                    for p in self.players)
                for w in self.gameweeks),
            name='budget_fh')
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (
                so.expr_sum(self.sell[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh')
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh')

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) ==
                so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks),
            name='equal_transfers')

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w] + self.free_transfers[w] -
                so.expr_sum(self.buy[p, w] for p in self.players) + 1 +
                self.hits[w]
                >= self.free_transfers[w + 1] for w in self.gameweeks),
            name='rolling_ft_rel')
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (
                10000 * (2 - self.free_transfers[w + 1]) >=
                self.hits[w] for w in self.gameweeks),
            name='4.42')
        # The minimum number of FT is 1
        self.model.add_constraints(
            (self.free_transfers[w + 1] >= 1 for w in self.gameweeks),
            name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (
                self.free_transfers[w + 1] <= 2 - self.wildcard[w] -
                self.freehit[w] for w in self.gameweeks),
            name='max_ft')

    def differential_model(
            self,
            nb_differentials=3,
            threshold=10,
            target='Top_100K'):
        """ Build a model that select differential players

        Args:
            nb_differentials (int): Number of differential players to include
            threshold (int): Percent after which a player is a differential
            target (str): Rank
        """
        self.data['Differential'] = np.where(
            self.data[target] < threshold, 1, 0)
        # A min numberof starter players must be differentials
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'Differential']
                    for p in self.players) >= nb_differentials
                for w in self.gameweeks),
            name='differentials')

    def biased_model(self, love, hate, hit_limit, two_ft_gw):
        """ Model where one can force players in and out

        Args:
            love (dict): Players to include: {(index, gw)}
            hate (dict): Players to exclude: {(index, gw)}
            hit_limit (dict): Number of hits: {(gw, amount)}: {(17, 5)}
            two_ft_gw (list): GW to have 2FT. ie. the (GW-1) where a FT is rolled.
        """
        for bias in love:
            if bias == 'buy' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['buy']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['buy']]), 'Player selected to buy does not exist.'
                # The forced-buy player must be bought
                self.model.add_constraints(
                    (
                        self.buy[p, w] == 1 for (p, w) in love[bias]
                        ),
                    name="force_buy")
            if bias == 'start' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['start']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['start']]), 'Player selected to start does not exist.'
                # The forced-in team player must be in the team
                self.model.add_constraints(
                    (
                        self.team[p, w] == 1 for (p, w) in love[bias]
                        ),
                    name="force_in")
            if bias == 'team' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['team']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['team']]), 'Player selected to be in the team does not exist.'
                # The forced-in starter player must be a starter
                self.model.add_constraints(
                    (
                        self.starter[p, w] == 1 for (p, w) in love[bias]
                        ),
                    name="force_starter")
            if bias == 'cap' and love[bias]:
                assert all([w in self.gameweeks for (_, w) in love['cap']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in love['cap']]), 'Player selected to be the captain does not exist.'
                # The forced-in cap player must be the captain
                self.model.add_constraints(
                    (
                        self.captain[p, w] == 1 for (p, w) in love[bias]
                        ),
                    name="force_captain")

        for bias in hate:
            if bias == 'sell' and hate[bias]:
                assert all([w in self.gameweeks for (_, w) in hate['sell']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in hate['sell']]), 'Player selected to sell does not exist.'
                # The forced-out player must be sold
                self.model.add_constraints(
                    (self.sell[p, w] == 1 for (p, w) in hate[bias]),
                    name="force_sell")
            if bias == 'bench' and hate[bias]:
                assert all([w in self.gameweeks for (_, w) in hate['bench']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in hate['bench']]), 'Player selected to start does not exist.'
                # The forced-out of starter player must not be starting
                self.model.add_constraints(
                    (self.starter[p, w] == 0 for (p, w) in hate[bias]),
                    name="force_bench")  # Force player out by a certain gw
            if bias == 'team' and hate[bias]:
                assert all([w in self.gameweeks for (_, w) in hate['team']]), 'Gameweek selected does not exist.'
                assert all([bias[0] in self.players for bias in hate['team']]), 'Player selected to be out of the team does not exist.'
                # The forced-out of team player must not be in team
                self.model.add_constraints(
                    (self.team[p, w] == 0 for (p, w) in hate[bias]),
                    name="force_out")

        for bias in hit_limit:
            if bias == 'max' and hit_limit[bias]:
                assert all([w in self.gameweeks for (w, _) in hit_limit['max']]), 'Gameweek selected does not exist.'
                # The number of hits under the maximum
                self.model.add_constraints(
                    (
                        self.hits[w] <= max_hit
                        for (w, max_hit) in hit_limit[bias]),
                    name='hits_max')
            if bias == 'eq' and hit_limit[bias]:
                assert all([w in self.gameweeks for (w, _) in hit_limit['eq']]), 'Gameweek selected does not exist.'
                # The number of hits equal to the choice
                self.model.add_constraints(
                    (
                        self.hits[w] == nb_hit
                        for (w, nb_hit) in hit_limit[bias]),
                    name='hits_eq')
            if bias == 'min' and hit_limit[bias]:
                assert all([w in self.gameweeks for (w, _) in hit_limit['min']]), 'Gameweek selected does not exist.'
                # The number of hits above the minumum
                self.model.add_constraints(
                    (
                        self.hits[w] >= min_hit
                        for (w, min_hit) in hit_limit[bias]),
                    name='hits_min')

        for gw in two_ft_gw:
            assert gw > self.start and gw <= self.start + self.horizon, 'Gameweek selected cannot be constrained.'
            # Force rolling free transfer
            self.model.add_constraint(
                self.free_transfers[gw] == 2,
                name=f'force_roll_{gw}')

    def automated_chips_model(
            self,
            objective_type='decay',
            decay_gameweek=0.9,
            vicecap_decay=0.1,
            decay_bench=[0.1, 0.1, 0.1, 0.1],
            ft_val=0,
            itb_val=0,
            hit_val=6,
            triple_val=12,
            bboost_val=14,
            freehit_val=18,
            wildcard_val=18):
        """ Build wildcard model for iteratively long horizon

        Args:
            objective_type (str): Decay to apply higher importance to early GW
                and Linear to apply uniform weights
            decay_gameweek (float): Weight decay per gameweek
            vicecap_decay (float): Weight applied to points scored by vice
            decay_bench (list): Weight applied to points scored by bench.
            ft_val (int): Value of rolling a transfer.
            itb_val (int): Value of having money in the bank.
            hit_val (int): Penalty of taking a hit.
            triple_val (int): Minumum expected added value of using this chip
            bboost_val (int): Minumum expected added value of using this chip
            freehit_val (int): Minumum expected added value of using this chip
            wildcard_val (int): Minumum expected added value of using this chip
        """
        # Model
        self.model = so.Model(name='auto_chips_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='team',
            vartype=so.binary)
        self.team_fh = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='team_fh',
            vartype=so.binary)
        self.starter = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='starter',
            vartype=so.binary)
        self.bench = self.model.add_variables(
            self.players,
            self.gameweeks,
            order,
            name='bench',
            vartype=so.binary)

        self.captain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='captain',
            vartype=so.binary)
        self.vicecaptain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='vicecaptain',
            vartype=so.binary)

        self.buy = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='buy',
            vartype=so.binary)
        self.sell = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='sell',
            vartype=so.binary)

        self.triple = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='3xc',
            vartype=so.binary)
        self.bboost = self.model.add_variables(
            self.gameweeks,
            name='bb',
            vartype=so.binary)
        self.freehit = self.model.add_variables(
            self.gameweeks,
            name='fh',
            vartype=so.binary)
        self.wildcard = self.model.add_variables(
            self.gameweeks,
            name='wc',
            vartype=so.binary)

        self.aux = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='aux',
            vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start-1, self.start+self.period+1),
            name='ft',
            vartype=so.integer,
            lb=0)
        self.hits = self.model.add_variables(
            self.all_gameweeks,
            name='hits',
            vartype=so.integer,
            lb=0)
        self.in_the_bank = self.model.add_variables(
            self.all_gameweeks,
            name='itb',
            vartype=so.continuous,
            lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.starter[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        cap = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.captain[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        vice = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                vicecap_decay * self.vicecaptain[p, w] *
                self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        bench = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                * self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        txc = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                2 * self.triple[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        hits = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (hit_val * self.hits[w]) for w in self.gameweeks)

        ftv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                ft_val * (self.free_transfers[w] - 1)
            ) for w in self.gameweeks[1:])

        itbv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                itb_val * self.in_the_bank[w]
            ) for w in self.gameweeks)

        triple_penalty = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (triple_val * so.expr_sum(self.triple[p, w] for p in self.players))
            for w in self.gameweeks)

        bboost_penalty = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (bboost_val * self.bboost[w]) for w in self.gameweeks)

        freehit_penalty = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (freehit_val * self.freehit[w]) for w in self.gameweeks)

        wildcard_penalty = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (wildcard_val * self.wildcard[w]) for w in self.gameweeks)

        self.model.set_objective(
            - starter - cap - vice - bench - txc - ftv - itbv + hits +
            triple_penalty + bboost_penalty + freehit_penalty + wildcard_penalty,
            name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints(
            (self.team[p, self.start - 1] == 1 for p in self.initial_team),
            name='initial_team')
        self.model.add_constraint(
            self.free_transfers[self.start] == self.rolling_transfer + 1,
            name='initial_ft')
        self.model.add_constraint(
            self.in_the_bank[self.start - 1] == self.bank,
            name='initial_itb')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <=
            -- (not self.threexc_used),
            name='tc_once')
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <=
            -- (not self.bboost_used),
            name='bb_once')
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <=
            -- (not self.freehit_used),
            name='fh_once')
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <=
            -- (not self.wildcard_used),
            name='wc_once')

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players) +
                self.bboost[w] + self.freehit[w] + self.wildcard[w] <= 1
                for w in self.gameweeks),
            name='chip_once')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 for w in self.gameweeks),
            name='gk_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='def_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='mid_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 for w in self.gameweeks),
            name='for_limit')

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit')

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 * self.freehit[w]
                for w in self.gameweeks),
            name='gk_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='def_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='mid_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 * self.freehit[w]
                for w in self.gameweeks),
            name='for_limit_fh')

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3 * self.freehit[w]
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit_fh')

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] for p in self.players) == 11 +
                4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'G']
                    for p in self.players) ==
                1 + self.bboost[w] for w in self.gameweeks),
            name='gk_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'D']
                    for p in self.players) >= 3 for w in self.gameweeks),
            name='def_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'M']
                    for p in self.players) >= 2 for w in self.gameweeks),
            name='mid_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'F']
                    for p in self.players) >= 1 for w in self.gameweeks),
            name='for_min')

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (
                self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.24')
        self.model.add_constraints(
            (
                self.aux[p, w] <= self.team[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.25')
        self.model.add_constraints(
            (
                self.aux[p, w] <= 1 - self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='4.26')

        # Captain
        # One captain (or one triple cap) must be picked once
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.captain[p, w] + self.triple[p, w]
                    for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain')
        # One vice captain must be picked once
        self.model.add_constraints(
            (
                so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1
                for w in self.gameweeks),
            name='one_vicecaptain')
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] +
                self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players for w in self.gameweeks),
            name='cap_in_starters')

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.bench[p, w, 0] for p in self.players
                    if self.data.loc[p, 'G'] == 1) <=
                1 for w in self.gameweeks),
            name='one_bench_gk')
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1
                for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot')

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='bench_team')
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players for w in self.gameweeks),
            name='bench_team_fh')

        # Budget
        sold_amount = {
            w: so.expr_sum(
                self.sell[p, w] * self.data.loc[p, 'SV'] for p in self.players)
            for w in self.gameweeks}
        bought_amount = {
            w: so.expr_sum(
                self.buy[p, w] * self.data.loc[p, 'BV'] for p in self.players)
            for w in self.gameweeks}
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] +
                sold_amount[w] - bought_amount[w] for w in self.gameweeks),
            name='budget')
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] ==
                self.team[p, w] for p in self.players for w in self.gameweeks),
            name='team_similarity')
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (
                self.sell[p, w] + self.buy[p, w] <= 1
                for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell')

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1] + so.expr_sum(
                    self.team[p, w - 1] * self.data.loc[p, 'SV']
                    for p in self.players) >=
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'BV']
                    for p in self.players)
                for w in self.gameweeks),
            name='budget_fh')
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (
                so.expr_sum(self.sell[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh')
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh')

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) ==
                so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks),
            name='equal_transfers')

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w] + self.free_transfers[w] -
                so.expr_sum(self.buy[p, w] for p in self.players) + 1 +
                self.hits[w]
                >= self.free_transfers[w + 1] for w in self.gameweeks),
            name='rolling_ft_rel')
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (
                10000 * (2 - self.free_transfers[w + 1]) >=
                self.hits[w] for w in self.gameweeks),
            name='4.42')
        # The minimum number of FT is 1
        self.model.add_constraints(
            (self.free_transfers[w + 1] >= 1 for w in self.gameweeks),
            name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (
                self.free_transfers[w + 1] <= 2 - self.wildcard[w] -
                self.freehit[w] for w in self.gameweeks),
            name='max_ft')

    def advanced_wildcard(
            self,
            freehit_gw=-1,
            bboost_gw=-1,
            threexc_gw=-1,
            objective_type='decay',
            decay_gameweek=[0.9, 0.8, 0.7],
            vicecap_decay=0.1,
            decay_bench=[0.1, 0.1, 0.1, 0.1],
            ft_val=0,
            itb_val=0,
            hit_val=6):
        """ Build wildcard model for iteratively long horizon

        Args:
            freehit_gw (int): Gw to use chip in
            bboost_gw (int): Gw to use chip in
            threexc_gw (int): Gw to use chip in
            objective_type (str): Decay to apply higher importance to early GW
                and Linear to apply uniform weights
            decay_gameweek (float): Weight decay per gameweek
            vicecap_decay (float): Weight applied to points scored by vice
            decay_bench (list): Weight applied to points scored by bench.
            ft_val (int): Value of rolling a transfer.
            itb_val (int): Value of having money in the bank.
            hit_val (int): Penalty of taking a hit.
        """
        assert (freehit_gw < self.horizon), "Select a GW within the horizon."
        assert (bboost_gw < self.horizon), "Select a GW within the horizon."
        assert (threexc_gw < self.horizon), "Select a GW within the horizon."

        assert not (self.freehit_used and freehit_gw >= 0), "Freehit chip was already used."
        assert not (self.bboost_used and bboost_gw >= 0), "Bench boost chip was already used."
        assert not (self.threexc_used and threexc_gw >= 0), "Tripple captain chip was already used."
        assert not self.wildcard_used, "Wildcard chip was already used."

        # Longterm Model
        model_name = 'longterm'
        self.model = so.Model(name=model_name + '_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='team',
            vartype=so.binary)
        self.team_fh = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='team_fh',
            vartype=so.binary)
        self.starter = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='starter',
            vartype=so.binary)
        self.bench = self.model.add_variables(
            self.players,
            self.gameweeks,
            order,
            name='bench',
            vartype=so.binary)

        self.captain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='captain',
            vartype=so.binary)
        self.vicecaptain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='vicecaptain',
            vartype=so.binary)

        self.buy = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='buy',
            vartype=so.binary)
        self.sell = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='sell',
            vartype=so.binary)

        self.triple = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='3xc',
            vartype=so.binary)
        self.bboost = self.model.add_variables(
            self.gameweeks,
            name='bb',
            vartype=so.binary)
        self.freehit = self.model.add_variables(
            self.gameweeks,
            name='fh',
            vartype=so.binary)
        self.wildcard = self.model.add_variables(
            self.gameweeks,
            name='wc',
            vartype=so.binary)

        self.aux = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='aux',
            vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start-1, self.start+self.period+1),
            name='ft',
            vartype=so.integer,
            lb=0)
        self.hits = self.model.add_variables(
            self.all_gameweeks,
            name='hits',
            vartype=so.integer,
            lb=0)
        self.in_the_bank = self.model.add_variables(
            self.all_gameweeks,
            name='itb',
            vartype=so.continuous,
            lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.starter[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        cap = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.captain[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        vice = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                vicecap_decay * self.vicecaptain[p, w] *
                self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        bench = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                * self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        txc = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                2 * self.triple[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        hits = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (hit_val * self.hits[w]) for w in self.gameweeks)

        itbv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                itb_val * self.in_the_bank[w]
            ) for w in self.gameweeks)

        self.model.set_objective(
            - starter - cap - vice - bench - txc - itbv + hits,
            name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints(
            (self.team[p, self.start - 1] == 1 for p in self.initial_team),
            name='initial_team')
        self.model.add_constraint(
            self.in_the_bank[self.start - 1] == self.bank,
            name='initial_itb')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <=
            -- (not self.threexc_used),
            name='tc_once')
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <=
            -- (not self.bboost_used),
            name='bb_once')
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <=
            -- (not self.freehit_used),
            name='fh_once')
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <=
            -- (not self.wildcard_used),
            name='wc_once')

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players) +
                self.bboost[w] + self.freehit[w] + self.wildcard[w] <= 1
                for w in self.gameweeks),
            name='chip_once')

        # The chips must be used on the selected GW
        # For printing
        self.model.add_constraint(
            (self.wildcard[self.start] == 1),
            name='wildcard_gw')

        if bboost_gw + 1:
            self.model.add_constraint(
                self.bboost[self.start + bboost_gw] == 1,
                name='bboost_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0,
                name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, self.start + threexc_gw] for p in self.players
                    ) == 1,
                name='triple_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, w] for p in self.players for w in self.gameweeks
                    ) == 0,
                name='triple_unused')

        if freehit_gw + 1:
            self.model.add_constraint(
                self.freehit[self.start + freehit_gw] == 1,
                name='freehit_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0,
                name='freehit_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 for w in self.gameweeks),
            name='gk_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='def_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='mid_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 for w in self.gameweeks),
            name='for_limit')

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit')

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 * self.freehit[w]
                for w in self.gameweeks),
            name='gk_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='def_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='mid_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 * self.freehit[w]
                for w in self.gameweeks),
            name='for_limit_fh')

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3 * self.freehit[w]
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit_fh')

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] for p in self.players) == 11 +
                4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'G']
                    for p in self.players) ==
                1 + self.bboost[w] for w in self.gameweeks),
            name='gk_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'D']
                    for p in self.players) >= 3 for w in self.gameweeks),
            name='def_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'M']
                    for p in self.players) >= 2 for w in self.gameweeks),
            name='mid_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'F']
                    for p in self.players) >= 1 for w in self.gameweeks),
            name='for_min')

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (
                self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.24')
        self.model.add_constraints(
            (
                self.aux[p, w] <= self.team[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.25')
        self.model.add_constraints(
            (
                self.aux[p, w] <= 1 - self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='4.26')

        # Captain
        # One captain (or one triple cap) must be picked
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.captain[p, w] + self.triple[p, w]
                    for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain')
        # One vice captain must be picked
        self.model.add_constraints(
            (
                so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1
                for w in self.gameweeks),
            name='one_vicecaptain')
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] +
                self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players for w in self.gameweeks),
            name='cap_in_starters')

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.bench[p, w, 0] for p in self.players
                    if self.data.loc[p, 'G'] == 1) <=
                1 for w in self.gameweeks),
            name='one_bench_gk')
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1
                for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot')

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='bench_team')
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players for w in self.gameweeks),
            name='bench_team_fh')

        # Budget
        sold_amount = {
            w: so.expr_sum(
                self.sell[p, w] * self.data.loc[p, 'SV'] for p in self.players)
            for w in self.gameweeks}
        bought_amount = {
            w: so.expr_sum(
                self.buy[p, w] * self.data.loc[p, 'BV'] for p in self.players)
            for w in self.gameweeks}
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] +
                sold_amount[w] - bought_amount[w] for w in self.gameweeks),
            name='budget')
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] ==
                self.team[p, w] for p in self.players for w in self.gameweeks),
            name='team_similarity')
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (
                self.sell[p, w] + self.buy[p, w] <= 1
                for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell')

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1] + so.expr_sum(
                    self.team[p, w - 1] * self.data.loc[p, 'SV']
                    for p in self.players) >=
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'BV']
                    for p in self.players)
                for w in self.gameweeks),
            name='budget_fh')
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (
                so.expr_sum(self.sell[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh')
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh')

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) ==
                so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks),
            name='equal_transfers')

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w] + self.free_transfers[w] -
                so.expr_sum(self.buy[p, w] for p in self.players) + 1 +
                self.hits[w]
                >= self.free_transfers[w + 1] for w in self.gameweeks),
            name='rolling_ft_rel')
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (
                10000 * (2 - self.free_transfers[w + 1]) >=
                self.hits[w] for w in self.gameweeks),
            name='4.42')
        # The minimum number of FT is 1
        self.model.add_constraints(
            (self.free_transfers[w + 1] >= 1 for w in self.gameweeks),
            name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (
                self.free_transfers[w + 1] <= 2 - self.wildcard[w] -
                self.freehit[w] for w in self.gameweeks),
            name='max_ft')

        # Enforce longterm planning by having no transfer past WC GW
        # NOTE: This messes up Rolling transfer logic but it matters not 
        # Since we're not using FTV in objective function.
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) == 0 for w in self.gameweeks[1:]),
            name=f'no_transfer')

        # Solve
        self.model.export_mps(filename=f"optimization/tmp/{model_name}.mps")
        command = (
            f'cbc optimization/tmp/{model_name}.mps solve solu ' +
            f'optimization/tmp/{model_name}_solution.txt')

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
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=model_name)

        # GW
        print("\n----------")
        # Get all players selected by longterm WC
        wc_team = [p for p in self.players if self.team[p, self.start].get_value()]

        # Medium range planning Model
        model_name = 'medium'
        self.model = so.Model(name=model_name + '_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='team',
            vartype=so.binary)
        self.team_fh = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='team_fh',
            vartype=so.binary)
        self.starter = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='starter',
            vartype=so.binary)
        self.bench = self.model.add_variables(
            self.players,
            self.gameweeks,
            order,
            name='bench',
            vartype=so.binary)

        self.captain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='captain',
            vartype=so.binary)
        self.vicecaptain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='vicecaptain',
            vartype=so.binary)

        self.buy = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='buy',
            vartype=so.binary)
        self.sell = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='sell',
            vartype=so.binary)

        self.triple = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='3xc',
            vartype=so.binary)
        self.bboost = self.model.add_variables(
            self.gameweeks,
            name='bb',
            vartype=so.binary)
        self.freehit = self.model.add_variables(
            self.gameweeks,
            name='fh',
            vartype=so.binary)
        self.wildcard = self.model.add_variables(
            self.gameweeks,
            name='wc',
            vartype=so.binary)

        self.aux = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='aux',
            vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start-1, self.start+self.period+1),
            name='ft',
            vartype=so.integer,
            lb=0)
        self.hits = self.model.add_variables(
            self.all_gameweeks,
            name='hits',
            vartype=so.integer,
            lb=0)
        self.in_the_bank = self.model.add_variables(
            self.all_gameweeks,
            name='itb',
            vartype=so.continuous,
            lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.starter[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        cap = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.captain[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        vice = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                vicecap_decay * self.vicecaptain[p, w] *
                self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        bench = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                * self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        txc = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                2 * self.triple[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        hits = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (hit_val * self.hits[w]) for w in self.gameweeks)

        ftv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                ft_val * (self.free_transfers[w] - 1)
            ) for w in self.gameweeks[1:])

        itbv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                itb_val * self.in_the_bank[w]
            ) for w in self.gameweeks)

        self.model.set_objective(
            - starter - cap - vice - bench - txc - ftv - itbv + hits,
            name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints(
            (self.team[p, self.start - 1] == 1 for p in self.initial_team),
            name='initial_team')
        self.model.add_constraint(
            self.in_the_bank[self.start - 1] == self.bank,
            name='initial_itb')
        self.model.add_constraint(
            (so.expr_sum(self.team[p, self.start] for p in wc_team) >= 10),
            name='initial_wc_team')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <=
            -- (not self.threexc_used),
            name='tc_once')
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <=
            -- (not self.bboost_used),
            name='bb_once')
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <=
            -- (not self.freehit_used),
            name='fh_once')
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <=
            -- (not self.wildcard_used),
            name='wc_once')

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players) +
                self.bboost[w] + self.freehit[w] + self.wildcard[w] <= 1
                for w in self.gameweeks),
            name='chip_once')

        # The chips must be used on the selected GW
        self.model.add_constraint(
            (self.wildcard[self.start] == 1),
            name='wildcard_gw')

        if bboost_gw + 1:
            self.model.add_constraint(
                self.bboost[self.start + bboost_gw] == 1,
                name='bboost_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0,
                name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, self.start + threexc_gw] for p in self.players
                    ) == 1,
                name='triple_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, w] for p in self.players for w in self.gameweeks
                    ) == 0,
                name='triple_unused')

        if freehit_gw + 1:
            self.model.add_constraint(
                self.freehit[self.start + freehit_gw] == 1,
                name='freehit_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0,
                name='freehit_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 for w in self.gameweeks),
            name='gk_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='def_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='mid_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 for w in self.gameweeks),
            name='for_limit')

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit')

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 * self.freehit[w]
                for w in self.gameweeks),
            name='gk_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='def_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='mid_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 * self.freehit[w]
                for w in self.gameweeks),
            name='for_limit_fh')

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3 * self.freehit[w]
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit_fh')

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] for p in self.players) == 11 +
                4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'G']
                    for p in self.players) ==
                1 + self.bboost[w] for w in self.gameweeks),
            name='gk_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'D']
                    for p in self.players) >= 3 for w in self.gameweeks),
            name='def_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'M']
                    for p in self.players) >= 2 for w in self.gameweeks),
            name='mid_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'F']
                    for p in self.players) >= 1 for w in self.gameweeks),
            name='for_min')

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (
                self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.24')
        self.model.add_constraints(
            (
                self.aux[p, w] <= self.team[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.25')
        self.model.add_constraints(
            (
                self.aux[p, w] <= 1 - self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='4.26')

        # Captain
        # One captain (or one triple cap) must be picked
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.captain[p, w] + self.triple[p, w]
                    for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain')
        # One vice captain must be picked
        self.model.add_constraints(
            (
                so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1
                for w in self.gameweeks),
            name='one_vicecaptain')
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] +
                self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players for w in self.gameweeks),
            name='cap_in_starters')

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.bench[p, w, 0] for p in self.players
                    if self.data.loc[p, 'G'] == 1) <=
                1 for w in self.gameweeks),
            name='one_bench_gk')
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1
                for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot')

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='bench_team')
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players for w in self.gameweeks),
            name='bench_team_fh')

        # Budget
        sold_amount = {
            w: so.expr_sum(
                self.sell[p, w] * self.data.loc[p, 'SV'] for p in self.players)
            for w in self.gameweeks}
        bought_amount = {
            w: so.expr_sum(
                self.buy[p, w] * self.data.loc[p, 'BV'] for p in self.players)
            for w in self.gameweeks}
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] +
                sold_amount[w] - bought_amount[w] for w in self.gameweeks),
            name='budget')
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] ==
                self.team[p, w] for p in self.players for w in self.gameweeks),
            name='team_similarity')
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (
                self.sell[p, w] + self.buy[p, w] <= 1
                for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell')

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1] + so.expr_sum(
                    self.team[p, w - 1] * self.data.loc[p, 'SV']
                    for p in self.players) >=
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'BV']
                    for p in self.players)
                for w in self.gameweeks),
            name='budget_fh')
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (
                so.expr_sum(self.sell[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh')
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh')

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) ==
                so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks),
            name='equal_transfers')

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w] + self.free_transfers[w] -
                so.expr_sum(self.buy[p, w] for p in self.players) + 1 +
                self.hits[w]
                >= self.free_transfers[w + 1] for w in self.gameweeks),
            name='rolling_ft_rel')
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (
                10000 * (2 - self.free_transfers[w + 1]) >=
                self.hits[w] for w in self.gameweeks),
            name='4.42')
        # The minimum number of FT is 1
        self.model.add_constraints(
            (self.free_transfers[w + 1] >= 1 for w in self.gameweeks),
            name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (
                self.free_transfers[w + 1] <= 2 - self.wildcard[w] -
                self.freehit[w] for w in self.gameweeks),
            name='max_ft')

        # Enforce medium term planning by having 1 transfer past WC GW
        self.model.add_constraints(
            (
                self.hits[w] <= 0
                for w in self.gameweeks),
            name='hits_max')

        # Solve
        self.model.export_mps(filename=f"optimization/tmp/{model_name}.mps")
        command = (
            f'cbc optimization/tmp/{model_name}.mps solve solu ' +
            f'optimization/tmp/{model_name}_solution.txt')

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
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=model_name)

        # GW
        print("\n----------")
        # Get all players selected by medium term WC
        wc_team = [p for p in self.players if self.team[p, self.start].get_value()]

        # Short range planning Model
        model_name = 'short'
        self.model = so.Model(name=model_name + '_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='team',
            vartype=so.binary)
        self.team_fh = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='team_fh',
            vartype=so.binary)
        self.starter = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='starter',
            vartype=so.binary)
        self.bench = self.model.add_variables(
            self.players,
            self.gameweeks,
            order,
            name='bench',
            vartype=so.binary)

        self.captain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='captain',
            vartype=so.binary)
        self.vicecaptain = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='vicecaptain',
            vartype=so.binary)

        self.buy = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='buy',
            vartype=so.binary)
        self.sell = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='sell',
            vartype=so.binary)

        self.triple = self.model.add_variables(
            self.players,
            self.gameweeks,
            name='3xc',
            vartype=so.binary)
        self.bboost = self.model.add_variables(
            self.gameweeks,
            name='bb',
            vartype=so.binary)
        self.freehit = self.model.add_variables(
            self.gameweeks,
            name='fh',
            vartype=so.binary)
        self.wildcard = self.model.add_variables(
            self.gameweeks,
            name='wc',
            vartype=so.binary)

        self.aux = self.model.add_variables(
            self.players,
            self.all_gameweeks,
            name='aux',
            vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start-1, self.start+self.period+1),
            name='ft',
            vartype=so.integer,
            lb=0)
        self.hits = self.model.add_variables(
            self.all_gameweeks,
            name='hits',
            vartype=so.integer,
            lb=0)
        self.in_the_bank = self.model.add_variables(
            self.all_gameweeks,
            name='itb',
            vartype=so.continuous,
            lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.starter[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        cap = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                self.captain[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        vice = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                vicecap_decay * self.vicecaptain[p, w] *
                self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        bench = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order)
                * self.data.loc[p, f'{w}_Pts'] for p in self.players)
            for w in self.gameweeks)

        txc = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            so.expr_sum(
                2 * self.triple[p, w] * self.data.loc[p, f'{w}_Pts']
                for p in self.players)
            for w in self.gameweeks)

        hits = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start)
                if objective_type == 'linear' else 1) *
            (hit_val * self.hits[w]) for w in self.gameweeks)

        ftv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                ft_val * (self.free_transfers[w] - 1)
            ) for w in self.gameweeks[1:])

        itbv = so.expr_sum(
            (
                np.power(decay_gameweek, w - self.start - 1)
                if objective_type == 'linear' else 1) *
            (
                itb_val * self.in_the_bank[w]
            ) for w in self.gameweeks)

        self.model.set_objective(
            - starter - cap - vice - bench - txc - ftv - itbv + hits,
            name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints(
            (self.team[p, self.start - 1] == 1 for p in self.initial_team),
            name='initial_team')
        self.model.add_constraint(
            self.in_the_bank[self.start - 1] == self.bank,
            name='initial_itb')
        self.model.add_constraint(
            (so.expr_sum(self.team[p, self.start] for p in wc_team) >= 13),
            name='initial_wc_team')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <=
            -- (not self.threexc_used),
            name='tc_once')
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <=
            -- (not self.bboost_used),
            name='bb_once')
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <=
            -- (not self.freehit_used),
            name='fh_once')
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <=
            -- (not self.wildcard_used),
            name='wc_once')

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players) +
                self.bboost[w] + self.freehit[w] + self.wildcard[w] <= 1
                for w in self.gameweeks),
            name='chip_once')

        # The chips must be used on the selected GW
        self.model.add_constraint(
            (self.wildcard[self.start] == 1),
            name='wildcard_gw')

        if bboost_gw + 1:
            self.model.add_constraint(
                self.bboost[self.start + bboost_gw] == 1,
                name='bboost_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0,
                name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, self.start + threexc_gw] for p in self.players
                    ) == 1,
                name='triple_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(
                    self.triple[p, w] for p in self.players for w in self.gameweeks
                    ) == 0,
                name='triple_unused')

        if freehit_gw + 1:
            self.model.add_constraint(
                self.freehit[self.start + freehit_gw] == 1,
                name='freehit_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0,
                name='freehit_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 for w in self.gameweeks),
            name='gk_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='def_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 for w in self.gameweeks),
            name='mid_limit')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 for w in self.gameweeks),
            name='for_limit')

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit')

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'G']
                    for p in self.players) == 2 * self.freehit[w]
                for w in self.gameweeks),
            name='gk_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'D']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='def_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'M']
                    for p in self.players) == 5 * self.freehit[w]
                for w in self.gameweeks),
            name='mid_limit_fh')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'F']
                    for p in self.players) == 3 * self.freehit[w]
                for w in self.gameweeks),
            name='for_limit_fh')

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name]
                    for p in self.players) <= 3 * self.freehit[w]
                for team_name in self.team_names for w in self.gameweeks),
            name='team_limit_fh')

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] for p in self.players) == 11 +
                4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'G']
                    for p in self.players) ==
                1 + self.bboost[w] for w in self.gameweeks),
            name='gk_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'D']
                    for p in self.players) >= 3 for w in self.gameweeks),
            name='def_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'M']
                    for p in self.players) >= 2 for w in self.gameweeks),
            name='mid_min')
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.starter[p, w] * self.data.loc[p, 'F']
                    for p in self.players) >= 1 for w in self.gameweeks),
            name='for_min')

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (
                self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.24')
        self.model.add_constraints(
            (
                self.aux[p, w] <= self.team[p, w]
                for p in self.players for w in self.gameweeks),
            name='4.25')
        self.model.add_constraints(
            (
                self.aux[p, w] <= 1 - self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='4.26')

        # Captain
        # One captain (or one triple cap) must be picked
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.captain[p, w] + self.triple[p, w]
                    for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain')
        # One vice captain must be picked
        self.model.add_constraints(
            (
                so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1
                for w in self.gameweeks),
            name='one_vicecaptain')
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] +
                self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players for w in self.gameweeks),
            name='cap_in_starters')

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(
                    self.bench[p, w, 0] for p in self.players
                    if self.data.loc[p, 'G'] == 1) <=
                1 for w in self.gameweeks),
            name='one_bench_gk')
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1
                for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot')

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players for w in self.gameweeks),
            name='bench_team')
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] +
                so.expr_sum(self.bench[p, w, o] for o in order) <=
                self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players for w in self.gameweeks),
            name='bench_team_fh')

        # Budget
        sold_amount = {
            w: so.expr_sum(
                self.sell[p, w] * self.data.loc[p, 'SV'] for p in self.players)
            for w in self.gameweeks}
        bought_amount = {
            w: so.expr_sum(
                self.buy[p, w] * self.data.loc[p, 'BV'] for p in self.players)
            for w in self.gameweeks}
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] +
                sold_amount[w] - bought_amount[w] for w in self.gameweeks),
            name='budget')
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] ==
                self.team[p, w] for p in self.players for w in self.gameweeks),
            name='team_similarity')
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (
                self.sell[p, w] + self.buy[p, w] <= 1
                for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell')

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1] + so.expr_sum(
                    self.team[p, w - 1] * self.data.loc[p, 'SV']
                    for p in self.players) >=
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, 'BV']
                    for p in self.players)
                for w in self.gameweeks),
            name='budget_fh')
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (
                so.expr_sum(self.sell[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh')
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) <=
                15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh')

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) ==
                so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks),
            name='equal_transfers')

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w] + self.free_transfers[w] -
                so.expr_sum(self.buy[p, w] for p in self.players) + 1 +
                self.hits[w]
                >= self.free_transfers[w + 1] for w in self.gameweeks),
            name='rolling_ft_rel')
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (
                10000 * (2 - self.free_transfers[w + 1]) >=
                self.hits[w] for w in self.gameweeks),
            name='4.42')
        # The minimum number of FT is 1
        self.model.add_constraints(
            (self.free_transfers[w + 1] >= 1 for w in self.gameweeks),
            name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (
                self.free_transfers[w + 1] <= 2 - self.wildcard[w] -
                self.freehit[w] for w in self.gameweeks),
            name='max_ft')

        # Solve
        self.model.export_mps(filename=f"optimization/tmp/{model_name}.mps")
        command = (
            f'cbc optimization/tmp/{model_name}.mps solve solu ' +
            f'optimization/tmp/{model_name}_solution.txt')

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

        return pretty_print(
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=model_name)

    def solve(self, model_name, log=False, i=0, time_lim=0):
        """ Solves the model

        Args:
            model_name (string): Model name
            log (bool): Sasoptpy logging progress
            i (int): Iteration (as part of suboptimals)
            time_lim (int): Time upper bound for the duration
                of optimization past the initial feasible solution
        """
        self.model.export_mps(filename=f"optimization/tmp/{model_name}.mps")
        if time_lim == 0:
            command = (
                f'cbc optimization/tmp/{model_name}.mps cost column solve solu ' +
                f'optimization/tmp/{model_name}_solution.txt')
            if log:
                os.system(command)
            else:
                process = Popen(command, shell=True, stdout=DEVNULL)
                process.wait()

        else:
            command = (
                f'cbc optimization/tmp/{model_name}.mps cost column ratio 1 solve solu ' +
                f'optimization/tmp/{model_name}_solution_feasible.txt')
            if log:
                os.system(command)
            else:
                process = Popen(command, shell=True, stdout=DEVNULL)
                process.wait()

            command = (
                f'cbc optimization/tmp/{model_name}.mps mips optimization/tmp/{model_name}_solution_feasible.txt ' +
                f'cost column sec {time_lim} solve solu optimization/tmp/{model_name}_solution.txt')
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

        return pretty_print(
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank, self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=i)

    def suboptimals(
            self,
            model_name,
            iterations=3,
            cutoff_search='first_transfer'):
        """ Solves model and gives suboptimal solution

        Args:
            model_name (string): Model name
            iterations (int): Iteration
            cutoff_search (str): Suboptimal solving method

        Returns:
            (dict): Dictionnary of hashes representing transfers
        """
        sa = {}

        for i in range(iterations):

            print(f"\n----- Solution {i+1} -----")
            self.solve(model_name + f'_{i}', i=i)

            if i != iterations - 1:
                # Select the players that have been transfered in/out
                if cutoff_search == 'first_buy':
                    actions = so.expr_sum(
                        self.buy[p, self.start] for p in self.players
                        if self.buy[p, self.start].get_value() > 0.5)
                    gw_range = [self.start]
                elif cutoff_search == 'horizon_buy':
                    actions = so.expr_sum(
                        so.expr_sum(
                            self.buy[p, w] for p in self.players
                            if self.buy[p, w].get_value() > 0.5)
                        for w in self.gameweeks)
                    gw_range = self.gameweeks
                elif cutoff_search == 'first_transfer':
                    actions = (
                        so.expr_sum(
                            self.buy[p, self.start] for p in self.players
                            if self.buy[p, self.start].get_value() > 0.5) +
                        so.expr_sum(
                            self.sell[p, self.start] for p in self.players
                            if self.sell[p, self.start].get_value() > 0.5)
                        )
                    gw_range = [self.start]
                elif cutoff_search == 'horizon_transfer':
                    actions = (
                        so.expr_sum(
                            so.expr_sum(
                                self.buy[p, w] for p in self.players
                                if self.buy[p, w].get_value() > 0.5)
                            for w in self.gameweeks) +
                        so.expr_sum(
                            so.expr_sum(
                                self.sell[p, w] for p in self.players
                                if self.sell[p, w].get_value() > 0.5)
                            for w in self.gameweeks)
                        )
                    gw_range = self.gameweeks

                if actions.get_value() != 0:
                    # This step forces one transfer to be unfeasible
                    # Note: the constraint is only applied to the activated
                    # transfers so the ones not activated are thus allowed.
                    self.model.add_constraint(
                        actions <= actions.get_value() - 1,
                        name=f'cutoff_{i}')
                else:
                    # Force one transfer in case of sub-optimal solution
                    # choosing to roll transfer
                    self.model.add_constraint(
                        so.expr_sum(
                            so.expr_sum(self.sell[p, w] for p in self.players)
                            for w in gw_range) >= 1,
                        name=f'cutoff_{i}')

            if self.freehit[self.start].get_value() :
                sa[i] = [
                    [
                        p for p in self.players
                        if self.team_fh[p, self.start].get_value() > 0.5],
                    [],
                    self.model.get_objective_value()
                ]
            else:
                sa[i] = [
                    [
                        p for p in self.players
                        if self.buy[p, self.start].get_value() > 0.5],
                    [
                        p for p in self.players
                        if self.sell[p, self.start].get_value() > 0.5],
                    self.model.get_objective_value()
                ]

        return sa

    def sensitivity_analysis(
            self,
            repeats=3,
            iterations=3,
            parameters={'model_name': 'sa'}):
        """ Solving model with randomized EV

        Args:
            repeats (int): Repeating the randomization/solving
            iterations (int): Iterations of suboptimals
        """
        podium = pd.DataFrame(columns=list(np.arange(1, 4)))
        hashes = {}
        raw_data = self.data.copy()

        # Reproduce the optimization from scratch
        for r in range(repeats):
            self.build_model(**parameters)

            print(f"\n----- Trial {r+1} -----")

            # Apply random noise to the original prediction
            # (i.e not stacking noise)
            self.data = raw_data.copy()
            self.random_noise(None)
            # Find optimal solutions
            sa = self.suboptimals(
                f"sensitivity_analysis_{r}",
                iterations=iterations)

            yield 1

            # Store data
            for i, (k, v) in enumerate(sa.items()):
                transfer = hash(tuple((tuple(v[0]), tuple(v[1]))))
                hashes[transfer] = v

                if transfer in podium.index:
                    podium.loc[transfer, i+1] += 1

                else:
                    for pos in range(1, iterations+1):
                        podium.loc[transfer, pos] = 0

                    podium.loc[transfer, i+1] = 1

                num_cols = sum([podium.loc[transfer, r] for r in np.arange(1, iterations+1)])
                podium.loc[
                    transfer,
                    f"EV_{int(num_cols)}"] = v[2]

        podium.fillna(0).to_csv("optimization/tmp/podium.csv")
        with open("optimization/tmp/hashes.json", "w") as outfile:
            json.dump(hashes, outfile)


if __name__ == "__main__":
    os.makedirs('optimization/tmp/', exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    to = Team_Optimization(
        team_id=30610,
        horizon=3,
        noise=False,
        premium=True)

    to.build_model(
        model_name="vanilla",
        freehit_gw=0,
        wildcard_gw=-1,
        bboost_gw=1,
        threexc_gw=-1,
        objective_type='decay',
        decay_gameweek=0.9,
        vicecap_decay=0.1,
        decay_bench=[0.03, 0.21, 0.06, 0.002],
        ft_val=1.5,
        itb_val=0.008,
        hit_val=6)

    # to.differential_model(
    #     nb_differentials=3,
    #     threshold=10,
    #     target='Top_100K')

    # to.automated_chips_model(
    #     objective_type='decay',
    #     decay_gameweek=0.9,
    #     vicecap_decay=0.1,
    #     decay_bench=[0.03, 0.21, 0.06, 0.002],
    #     ft_val=1.5,
    #     itb_val=0.008,
    #     hit_val=6)

    # to.advanced_wildcard(
    #     freehit_gw=-1,
    #     bboost_gw=-1,
    #     threexc_gw=-1,
    #     objective_type='decay',
    #     decay_gameweek=0.9,
    #     vicecap_decay=0.1,
    #     decay_bench=[0.03, 0.21, 0.06, 0.002],
    #     ft_val=1.5,
    #     itb_val=0.008,
    #     hit_val=6)

    # to.biased_model(
    #     love={
    #         'buy': {},
    #         'start': {},
    #         'team': {},
    #         'cap': {}
    #     },
    #     hate={
    #         'sell': {},
    #         'team': {},
    #         'bench': {}
    #     },
    #     hit_limit={
    #         'max': {},
    #         'eq': {},
    #         'min': {}
    #     },
    #     two_ft_gw=[])

    to.solve(
        model_name="vanilla",
        log=True,
        time_lim=0)

    # tp.suboptimals(
    #     model_name="vanilla",
    #     iterations=3,
    #     cutoff_search='first_transfer')

    # tp.sensitivity_analysis(
    #     repeats=2,
    #     iterations=3)
