import streamlit as st

import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from highlight_text import fig_text
from matplotlib.colors import ListedColormap

from team_optimization import Team_Optimization


def write():
    st.title('FPL - Sensitivity Analysis Model')
    st.header(
        """
        Sensitivity Analysis FPL Optimization.
        """)

    plt.style.use(".streamlit/style.mplstyle")

    with st.expander('Parameters', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Horizon", min_value=1, max_value=8, value=5, step=1)
        with col2:
            premium = st.selectbox("Data type", ['Premium', 'Free'], 0)


        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gk_weight = st.slider("GK Weight", min_value=0.01, max_value=1., value=0.03, step=0.02)
        with col2:
            first_bench_weight = st.slider("1st Weight", min_value=0.01, max_value=1., value=0.21, step=0.02)
        with col3:
            second_bench_weight = st.slider("2nd Weight", min_value=0.01, max_value=1., value=0.06, step=0.02)
        with col4:
            third_bench_weight = st.slider("3rd Weight", min_value=0.01, max_value=1., value=0.01, step=0.02)


        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            decay = st.slider("Decay rate", min_value=0., max_value=1., value=0.9, step=0.02)
        with col2:
            vicecap_decay = st.slider("Vicecap rate", min_value=0., max_value=1., value=0.1, step=0.02)
        with col3:
            ft_val = st.slider("FT value", min_value=0., max_value=5., value=1.5, step=0.2)
        with col4:
            hit_val = st.slider("Hit value", min_value=2., max_value=8., value=6., step=0.5)
        with col5:
            itb_val = st.slider("ITB value", min_value=0., max_value=1., value=0.008, step=0.02)


        col1, col2 = st.columns(2)
        with col1:
            repeats = st.slider("Number of Experiments", min_value=1, max_value=10, value=5)
        with col2:
            iterations = st.slider("Iterations per exp.", min_value=1, max_value=10, value=7)


    if st.button('Run Optimization'):

        with st.spinner("Running Optimization ..."):
            to = Team_Optimization(
                team_id=35868,
                horizon=horizon,
                noise=False,
                premium=True if premium=='Premium' else False)

            to.build_model(
                model_name="vanilla",
                objective_type='decay' if decay != 0 else 'linear',
                decay_gameweek=decay,
                vicecap_decay=vicecap_decay,
                decay_bench=[gk_weight, first_bench_weight, second_bench_weight, third_bench_weight],
                ft_val=ft_val,
                itb_val=itb_val,
                hit_val=hit_val)

            tp.sensitivity_analysis(
                repeats=repeats,
                iterations=iterations)

            player_names = (
                pd
                .read_csv('data/fpl_official/vaastav/data/2021-22/player_idlist.csv')
                .set_index('id'))

            with open('optimization/tmp/hashes.json', 'r') as f:
                hashes = json.load(f)

            df = pd.read_csv("optimization/tmp/podium.csv")

            df['Total'] = df.apply(
                lambda x: sum(x[str(col)] for col in np.arange(1, iterations+1)),
                axis=1)

            (
                df['Transfer in'],
                df['Transfer out']) = zip(
                    *df['Unnamed: 0'].apply(
                        lambda x: (hashes[str(x)][0], hashes[str(x)][1])))
            df['Transfer'] = (
                df[['Transfer in', 'Transfer out']]
                .apply(lambda x: write_transfer(x, player_names), axis=1))

            max_cols = [col for col in df.columns if 'EV_' in col]

            df['Mean'] = (
                df.apply(
                    lambda x: - sum(x[col] if pd.notnull(x[col]) else 0 for col in max_cols) / x["Total"],
                    axis=1))
            df['Std'] = (
                df.apply(
                    lambda x: np.sqrt(
                        np.sum(
                            np.power(
                                [x[col] + x['Mean'] if pd.notnull(x[col]) else 0 for col in max_cols],
                                2)
                            ) / x["Total"]),
                    axis=1))

            df[['Total', '1', '2', '3']] = df[['Total', '1', '2', '3']].astype('int32')
            # df = df.sort_values(['Total', '1', '2', '3'], ascending=False)
            # st.dataframe(df[['Transfer', 'Total', '1', '2', '3', 'Mean', 'Std']])

            fig, ax = plt.subplots(figsize=(16, 12))
            # Set up the axis limits with a bit of padding
            ax.set_ylim(0, 5.5)
            ax.set_xlim(0, 25+(4+1)*16+2.5)
            ax.axis('off')
            header_pos = 5.25

            newaxes = []
            for j, row in df.head(7)[::-1].reset_index().iterrows():
                rectangle = patches.Rectangle(
                    (0, j*.75),
                    25, .5,
                    facecolor='#656B73')
                ax.add_patch(rectangle)
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width()/2.0
                cy = ry + rectangle.get_height()/2.0
                ax.annotate(
                    row['Transfer'],
                    (cx, cy),
                    color='w',
                    weight='bold',
                    fontsize=14,
                    ha='center',
                    va='center')

                newax = fig.add_axes([.63, 0.11+j*.105, .25 , .07])
                newax.boxplot(
                    [-row[col] for col in max_cols if pd.notnull(row[col])],
                    vert=False,
                    patch_artist=True,
                    capprops=dict(color='#656B73'),
                    boxprops=dict(facecolor='#656B73', color='#656B73'),
                    whiskerprops=dict(color='#656B73'),
                    flierprops=dict(markerfacecolor='r'),
                    medianprops=dict(color='w'),
                    widths=.6)

                newax.set_xlim(
                    -np.max(np.max(df[max_cols], axis=0))-1,
                    -np.min(np.min(df[max_cols], axis=0))+1)
                newaxes.append(newax)

            # Column headers
            ax.text(
                cx, header_pos,
                'Transfer',
                fontsize=16, weight='bold', ha='center')

            # Header separator
            ax.plot(
                [0, 25],
                [5.125, 5.125],
                ls='-', lw='2.5', c='grey')

            for i, col in enumerate(['Total', '1', '2', '3']):
                for j, row in df.head(7)[::-1].reset_index().iterrows():
                    rectangle = patches.Rectangle(
                        (30+i*10, j*.75),
                        5, .5,
                        facecolor='#656B73')
                    ax.add_patch(rectangle)
                    rx, ry = rectangle.get_xy()
                    cx = rx + rectangle.get_width()/2.0
                    cy = ry + rectangle.get_height()/2.0
                    ax.annotate(
                        row[col],
                        (cx, cy),
                        color='w',
                        weight='bold',
                        fontsize=14,
                        ha='center',
                        va='center')

                # Column headers
                ax.text(
                    cx, header_pos,
                    col,
                    fontsize=16, weight='bold', ha='center')

                # Header separator
                ax.plot(
                    [30+i*10, 30+i*10+5],
                    [5.125, 5.125],
                    ls='-', lw='2.5', c='grey')


            st.pyplot(fig, ax)
            plt.close(fig)


def write_transfer(x, player_names):
    if len(x['Transfer in']) == len(x['Transfer out']) == 0:
        return 'Roll Transfer'

    else:
        in_list, out_list = [], []
        for transfer_out in x['Transfer out']:
            out_list.append(player_names.loc[transfer_out]['second_name'])


        for transfer_in in x['Transfer in']:
            in_list.append(player_names.loc[transfer_in]['second_name'])

        return " + ".join(out_list) + " -> " + " + ".join(in_list)
