import streamlit as st

import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from team_optimization import Team_Optimization

@st.cache
def get_data():

    to = Team_Optimization(
        team_id=35868,
        horizon=5,
        noise=False,
        premium=True)

    return to.data.Name, to.start

def write():
    st.title('FPL - Sensitivity Analysis Model')
    st.header(
        """
        Sensitivity Analysis FPL Optimization.
        """)

    plt.style.use(".streamlit/style.mplstyle")
    player_names, start = get_data()

    with st.expander('Basics'):

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


    with st.expander('Chip selection'):

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            wc_gw = st.selectbox("Wildcard", [None] + [gw for gw in np.arange(horizon)], 0)
        with col2:
            fh_gw = st.selectbox("Freehit", [None] + [gw for gw in np.arange(horizon)], 0)
        with col3:
            tc_gw = st.selectbox("Triple Captain", [None] + [gw for gw in np.arange(horizon)], 0)
        with col4:
            bb_gw = st.selectbox("Bench Boost", [None] + [gw for gw in np.arange(horizon)], 0)


    with st.expander('Parameters', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            repeats = st.slider("Number of Experiments", min_value=1, max_value=25, value=5)
        with col2:
            iterations = st.slider("Iterations per exp.", min_value=1, max_value=25, value=7)


    if st.button('Run Optimization'):

        with st.spinner("Running Optimization ..."):
            if fh_gw == 0 and (horizon > 1 or iterations > 1):
                st.warning('This is a warning')

            else:
                # to = Team_Optimization(
                #     team_id=35868,
                #     horizon=horizon,
                #     noise=False,
                #     premium=True if premium=='Premium' else False)

                # to.sensitivity_analysis(
                #     repeats=repeats,
                #     iterations=iterations,
                #     parameters={
                #         'model_name':'sensitivity_analysis',
                #         'freehit_gw':fh_gw if fh_gw is not None else -1,
                #         'wildcard_gw':wc_gw if wc_gw is not None else -1,
                #         'bboost_gw':bb_gw if bb_gw is not None else -1,
                #         'threexc_gw':tc_gw if tc_gw is not None else -1,
                #         'objective_type':'decay' if decay != 0 else 'linear',
                #         'decay_gameweek':decay,
                #         'vicecap_decay':vicecap_decay,
                #         'decay_bench':[gk_weight, first_bench_weight, second_bench_weight, third_bench_weight],
                #         'ft_val':ft_val,
                #         'itb_val':itb_val,
                #         'hit_val':hit_val
                #     })

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

                if fh_gw == 0:
                    freehit_teams = pd.DataFrame(
                        zip(*df['Transfer in'].apply(lambda x: write_freehit(x, player_names)))
                        ).T

                    percent = pd.DataFrame(columns=['Player', 'Pos', 'Appearences'])
                    percent['Player'] = np.unique(freehit_teams)
                    player_pos = (
                        pd
                        .read_csv('data/fpl_official/vaastav/data/2021-22/cleaned_players.csv')
                        [['first_name', 'second_name', 'element_type']])
                    percent['Pos'] = (
                        percent['Player']
                        .apply(
                            lambda x: player_pos.loc[
                                player_pos.first_name + ' ' + player_pos.second_name == x
                                ]['element_type'].values[0]))
                    percent['Appearences'] = percent['Player'].apply(
                        lambda x: np.sum(np.sum(freehit_teams == x)))

                    percent['Mean'] = percent.apply(
                        lambda x: np.sum(np.sum(freehit_teams == x['Player'], axis=1)*df['Mean'])/x['Appearences'],
                        axis=1)
                    percent['Std'] = percent.apply(
                        lambda x: np.sum(np.sum(freehit_teams == x['Player'], axis=1)*df['Std'])/x['Appearences'],
                        axis=1)

                    with st.expander("Percentages", expanded=False):
                        st.write('Goalkeepers')
                        st.dataframe(
                            percent
                            .loc[percent.Pos == 'GK']
                            .sort_values(
                                by=['Appearences', 'Mean'],
                                ascending=[False, False])
                            [['Player', 'Appearences', 'Mean', 'Std']]
                            .reset_index(drop=True)
                            )

                        st.write('Defenders')
                        st.dataframe(
                            percent
                            .loc[percent.Pos == 'DEF']
                            .sort_values(
                                by=['Appearences', 'Mean'],
                                ascending=[False, False])
                            [['Player', 'Appearences', 'Mean', 'Std']]
                            .reset_index(drop=True)
                            )

                        st.write('Midfielders')
                        st.dataframe(
                            percent
                            .loc[percent.Pos == 'MID']
                            .sort_values(
                                by=['Appearences', 'Mean'],
                                ascending=[False, False])
                            [['Player', 'Appearences', 'Mean', 'Std']]
                            .reset_index(drop=True)
                            )

                        st.write('Forwards')
                        st.dataframe(
                            percent
                            .loc[percent.Pos == 'FWD']
                            .sort_values(
                                by=['Appearences', 'Mean'],
                                ascending=[False, False])
                            [['Player', 'Appearences', 'Mean', 'Std']]
                            .reset_index(drop=True)
                            )

                    fig, ax = plt.subplots(figsize=(16, 12))
                    ax.set_ylim(0, 4.5)
                    ax.set_xlim(0, 100)
                    ax.axis('off')

                    # Goalkeeper
                    df = (
                        percent
                        .loc[percent.Pos == 'GK']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        [['Player', 'Appearences', 'Mean', 'Std']]
                        .reset_index(drop=True)
                        .head(2))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (25+j*35, 3.5),
                            17, .5,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        ax.annotate(
                            row['Player'] + '\n\n' + str(row['Appearences']),
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=14,
                            ha='center',
                            va='center')

                    # Defender
                    df = (
                        percent
                        .loc[percent.Pos == 'DEF']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        [['Player', 'Appearences', 'Mean', 'Std']]
                        .reset_index(drop=True)
                        .head(5))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (2.5+j*20, 2.5),
                            17, .5,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        ax.annotate(
                            row['Player'] + '\n\n' + str(row['Appearences']),
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=14,
                            ha='center',
                            va='center')

                    # Midfielder
                    df = (
                        percent.loc[percent.Pos == 'MID']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        [['Player', 'Appearences', 'Mean', 'Std']]
                        .reset_index(drop=True)
                        .head(5))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (2.5+j*20, 1.5),
                            17, .5,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        name = row['Player'] if row['Player'] != "Bruno Miguel Borges Fernandes" else "Bruno Fernandes"
                        ax.annotate(
                            name + '\n\n' + str(row['Appearences']),
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=14,
                            ha='center',
                            va='center')

                    # Forward
                    df = (
                        percent.loc[percent.Pos == 'FWD']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        [['Player', 'Appearences', 'Mean', 'Std']]
                        .reset_index(drop=True)
                        .head(3))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (17.5+j*25, .5),
                            17, .5,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        name = row['Player'] if row['Player'] != "Cristiano Ronaldo dos Santos Aveiro" else "Cristiano Ronaldo"
                        ax.annotate(
                            name + '\n\n' + str(row['Appearences']),
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=14,
                            ha='center',
                            va='center')

                    st.pyplot(fig, ax)
                    plt.close(fig)

                else:
                    df['Transfer'] = (
                        df[['Transfer in', 'Transfer out']]
                        .apply(lambda x: write_transfer(x, player_names), axis=1))

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


def write_freehit(x, player_names):
    fh_list = []

    for fh_player in x:
        fh_list.append(
            player_names.loc[fh_player]['first_name'] +
            ' ' +
            player_names.loc[fh_player]['second_name'])

    return fh_list
