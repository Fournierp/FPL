import pandas as pd
import os
import sasoptpy as so


# Using the predicted points from https://fplreview.com/
filepath = "../data/fpl_review/2021-22/gameweek/7/fplreview_fp.csv"
# Gameweek of interest
gw = '7_Pts'
budget = 100
# Get the goalkeeper data
df = pd.read_csv(filepath)
gk_data = df[df['Pos'] == 1].copy().reset_index()
gk_data.set_index('index', inplace=True)
goalkeepers = gk_data.index.tolist()

# Create a model
model = so.Model(name='goalkeeper_model')

# Define variables
lineup = model.add_variables(goalkeepers, name='lineup', vartype=so.binary)
bench = model.add_variables(goalkeepers, name='bench', vartype=so.binary)

# Define Objective: maximize total expected points
total_xp = so.expr_sum(lineup[p] * gk_data.loc[p, gw] for p in goalkeepers) + \
           0.1 * so.expr_sum(bench[p] * gk_data.loc[p, gw] for p in goalkeepers)
model.set_objective(-total_xp, name='total_xp_obj', sense='N')

# Define Constraints
# The cost of the squad must exceed the budget
model.add_constraint(so.expr_sum((lineup[p] + bench[p]) * gk_data.loc[p, 'BV'] for p in goalkeepers) <= budget,
                     name='budget')

# The number of keeper must be 1 on bench and field
model.add_constraint(so.expr_sum(lineup[p] for p in goalkeepers) == 1, name='single_starter')
model.add_constraint(so.expr_sum(bench[p] for p in goalkeepers) == 1, name='single_substitute')

# A goalkeeper must not be picked more than once
model.add_constraints((lineup[p] + bench[p] <= 1 for p in goalkeepers), name='lineup_or_bench')

# Solve Step
model.export_mps(filename='goalkeeper.mps')
command = 'cbc goalkeeper.mps solve solu goalkeeper_solution.txt'
# !{command}
os.system(command)
with open('goalkeeper_solution.txt', 'r') as f:
    for v in model.get_variables():
        v.set_value(0)
    for line in f:
        if 'objective value' in line:
            continue
        words = line.split()
        var = model.get_variable(words[1])
        var.set_value(float(words[2]))
print("LINEUP")
for p in goalkeepers:
    if lineup[p].get_value() > 0.5:
        print(p, gk_data.loc[p])
print("BENCH")
for p in goalkeepers:
    if bench[p].get_value() > 0.5:
        print(p, gk_data.loc[p])
