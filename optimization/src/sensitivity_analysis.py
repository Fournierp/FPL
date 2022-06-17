import streamlit as st

import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from highlight_text import fig_text

from team_optimization import Team_Optimization

@st.cache
def get_data():
    """ Get player names, xpts for first GW and next GW number

    Returns:
        tuple: Series, int, series
    """
    to = Team_Optimization(
        team_id=35868,
        horizon=5,
        noise=False,
        premium=True)

    return to.data.Name, to.start, to.data[[f'{to.start}_Pts']]

def write():
    st.title('FPL - Sensitivity Analysis Model')
    st.header(
        """
        Sensitivity Analysis FPL Optimization.
        """)

    plt.style.use(".streamlit/style.mplstyle")
    player_names, start, xpts = get_data()


    with st.expander('Basics'):

        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Horizon", min_value=1, max_value=min(39-start, 8), value=min(39-start, 5), step=1)
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
            wc_gw = st.selectbox("Wildcard", [None] + [start + gw for gw in np.arange(horizon)], 0)
        with col2:
            fh_gw = st.selectbox("Freehit", [None] + [start + gw for gw in np.arange(horizon)], 0)
        with col3:
            tc_gw = st.selectbox("Triple Captain", [None] + [start + gw for gw in np.arange(horizon)], 0)
        with col4:
            bb_gw = st.selectbox("Bench Boost", [None] + [start + gw for gw in np.arange(horizon)], 0)


    with st.expander('Parameters', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            repeats = st.slider("Number of Experiments", min_value=1, max_value=25, value=5)
        with col2:
            iterations = st.slider("Iterations per exp.", min_value=1, max_value=25, value=7)


    if st.button('Run Optimization'):

        with st.spinner("Running Optimization ..."):
            if fh_gw == 0 and (horizon > 1 or iterations > 1):
                st.warning('Should not have more than 1 iteration or longer than 1 gw horizon')

            else:
                to = Team_Optimization(
                    team_id=35868,
                    horizon=horizon,
                    noise=False,
                    premium=True if premium=='Premium' else False)

                my_bar = st.progress(0)
                progress = 0

                for tmp in to.sensitivity_analysis(
                        repeats=repeats,
                        iterations=iterations,
                        parameters={
                            'model_name':'sensitivity_analysis',
                            'freehit_gw':fh_gw-start if fh_gw is not None else -1,
                            'wildcard_gw':wc_gw-start if wc_gw is not None else -1,
                            'bboost_gw':bb_gw-start if bb_gw is not None else -1,
                            'threexc_gw':tc_gw-start if tc_gw is not None else -1,
                            'objective_type':'decay' if decay != 0 else 'linear',
                            'decay_gameweek':decay,
                            'vicecap_decay':vicecap_decay,
                            'decay_bench':[gk_weight, first_bench_weight, second_bench_weight, third_bench_weight],
                            'ft_val':ft_val,
                            'itb_val':itb_val,
                            'hit_val':hit_val
                        }):

                    progress += 1
                    my_bar.progress(progress / repeats)

                player_names = (
                    pd
                    .read_csv('data/fpl_official/vaastav/data/2021-22/player_idlist.csv')
                    .set_index('id'))

                # Get first GW optimal teams
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
                evs = np.unique(df[max_cols].values)

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

                if fh_gw == 0 or wc_gw == 0:
                    # Freehit graph

                    # Freehit lineups
                    freehit_teams = pd.DataFrame(
                        zip(*df['Transfer in'].apply(lambda x: write_freehit(x, player_names)))
                        ).T

                    # Gather data on the players in the team
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

                    # Get team names
                    photos = (
                        pd.read_csv('data/fpl_official/vaastav/data/2021-22/players_raw.csv')
                        [['first_name', 'second_name', 'photo', 'team']])
                    url = "https://resources.premierleague.com/premierleague/photos/players/110x140/p{index}.png"

                    percent['Team'] = percent.apply(
                        lambda x: photos.loc[
                            photos.first_name + ' ' + photos.second_name == x.Player
                            ]['team'].values[0],
                        axis=1)

                    percent = pd.merge(
                        percent,
                        pd.read_csv('data/fpl_official/vaastav/data/2021-22/teams.csv')[['id', 'short_name']],
                        left_on='Team',
                        right_on='id',
                    ).drop(['Team', 'id'], axis=1)

                    # Add upcoming fixtures
                    fixtures = pd.read_csv('data/fpl_official/vaastav/data/2021-22/fixtures.csv')
                    fixtures = fixtures.loc[fixtures.event == start][['team_h', 'team_a']]
                    fixtures = pd.merge(
                        fixtures,
                        pd.read_csv('data/fpl_official/vaastav/data/2021-22/teams.csv')[['id', 'short_name']],
                        left_on='team_h',
                        right_on='id',
                    ).drop(['team_h', 'id'], axis=1).rename(columns={'short_name': 'team_h'})

                    fixtures = pd.merge(
                        fixtures,
                        pd.read_csv('data/fpl_official/vaastav/data/2021-22/teams.csv')[['id', 'short_name']],
                        left_on='team_a',
                        right_on='id',
                    ).drop(['team_a', 'id'], axis=1).rename(columns={'short_name': 'team_a'})

                    def get_fixtures(x, y):
                        if y.team_h == x:
                            return y.team_a

                        if y.team_a == x:
                            return y.team_h.lower()

                    percent['GW'] = percent['short_name'].apply(
                        lambda x: ' + '.join(
                            [
                                game for game in fixtures.apply(
                                    lambda y: get_fixtures(x, y),
                                    axis=1
                                ).values if game is not None]
                            )
                        )

                    # Add xpts data
                    xpts = pd.merge(
                        xpts,
                        player_names,
                        left_index=True,
                        right_index=True,
                    )
                    xpts['Player'] = xpts.apply(lambda x: x.first_name + ' ' + x.second_name, axis=1)

                    percent = pd.merge(
                        percent,
                        xpts,
                        left_on='Player',
                        right_on='Player'
                    )

                    fig, ax = plt.subplots(figsize=(12, 14))

                    # Goalkeeper
                    df = (
                        percent
                        .loc[percent.Pos == 'GK']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        .reset_index(drop=True)
                        .head(2))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (25+j*35, 3),
                            17, .25,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        name = row['Player'] if '-' not in row['Player'] else row['Player'].split(' ')[-1]
                        ax.annotate(
                            f"{name}\n{row['GW']}\n{np.round(row['Appearences']/repeats*100)} | xPts: {row[f'{start}_Pts']}",
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=11,
                            ha='center',
                            va='center')
                        # Plot the portrait
                        imscatter(
                            x=cx, y=3.25+.31,
                            image=url.format(index=(
                                photos.loc[
                                    photos.first_name + ' ' +
                                    photos.second_name == row['Player']
                                ]['photo'].values[0][:-4])),
                            ax=ax, zoom=.4)

                    # Defender
                    df = (
                        percent
                        .loc[percent.Pos == 'DEF']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        .reset_index(drop=True)
                        .head(5))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (2.5+j*20, 2),
                            17, .25,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        name = row['Player'] if '-' not in row['Player'] else row['Player'].split(' ')[-1]
                        ax.annotate(
                            # name + '\n' + row['GW'] + '\n' + str(row['Appearences']),
                            f"{name}\n{row['GW']}\n{np.round(row['Appearences']/repeats*100)} | xPts: {row[f'{start}_Pts']}",
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=11,
                            ha='center',
                            va='center')
                        # Plot the portrait
                        imscatter(
                            x=cx, y=2.25+.31,
                            image=url.format(index=(
                                photos.loc[
                                    photos.first_name + ' ' +
                                    photos.second_name == row['Player']
                                ]['photo'].values[0][:-4])),
                            ax=ax, zoom=.4)

                    # Midfielder
                    df = (
                        percent.loc[percent.Pos == 'MID']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        .reset_index(drop=True)
                        .head(5))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (2.5+j*20, 1),
                            17, .25,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        name = row['Player'] if row['Player'] != "Bruno Miguel Borges Fernandes" else "Bruno Fernandes"
                        name = name if '-' not in name else name.split(' ')[-1]
                        ax.annotate(
                            # name + '\n' + row['GW'] + '\n' + str(row['Appearences']),
                            f"{name}\n{row['GW']}\n{np.round(row['Appearences']/repeats*100)} | xPts: {row[f'{start}_Pts']}",
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=11,
                            ha='center',
                            va='center')
                        # Plot the portrait
                        imscatter(
                            x=cx, y=1.25+.31,
                            image=url.format(index=(
                                photos.loc[
                                    photos.first_name + ' ' +
                                    photos.second_name == row['Player']
                                ]['photo'].values[0][:-4])),
                            ax=ax, zoom=.4)

                    # Forward
                    df = (
                        percent.loc[percent.Pos == 'FWD']
                        .sort_values(
                            by=['Appearences', 'Mean'],
                            ascending=[False, False])
                        .reset_index(drop=True)
                        .head(3))

                    for j, row in df.iterrows():
                        rectangle = patches.Rectangle(
                            (17.5+j*25, 0),
                            17, .25,
                            facecolor='#656B73')
                        ax.add_patch(rectangle)
                        rx, ry = rectangle.get_xy()
                        cx = rx + rectangle.get_width()/2.0
                        cy = ry + rectangle.get_height()/2.0
                        name = row['Player'] if row['Player'] != "Cristiano Ronaldo dos Santos Aveiro" else "Cristiano Ronaldo"
                        name = name if '-' not in name else name.split(' ')[-1]
                        ax.annotate(
                            # name + '\n' + row['GW'] + '\n' + str(row['Appearences']),
                            f"{name}\n{row['GW']}\n{np.round(row['Appearences']/repeats*100)} | xPts: {row[f'{start}_Pts']}",
                            (cx, cy),
                            color='w',
                            weight='bold',
                            fontsize=11,
                            ha='center',
                            va='center')
                        # Plot the portrait
                        imscatter(
                            x=cx, y=.25+.31,
                            image=url.format(index=(
                                photos.loc[
                                    photos.first_name + ' ' +
                                    photos.second_name == row['Player']
                                ]['photo'].values[0][:-4])),
                            ax=ax, zoom=.4)

                    ax.set_ylim(0, 4.25)
                    ax.set_xlim(0, 100)
                    ax.axis('off')

                    fig_text(
                        x=0.14, y=.855,
                        s="<Players appearing most on randomized FH Team>",
                        highlight_textprops=[{"fontweight": "bold"}],
                        fontsize=24, fontfamily="DejaVu Sans", color='w')

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
                            [-row[col] for col in max_cols if pd.notnull(row[col]) and row[col] != 0],
                            vert=False,
                            patch_artist=True,
                            capprops=dict(color='#656B73'),
                            boxprops=dict(facecolor='#656B73', color='#656B73'),
                            whiskerprops=dict(color='#656B73'),
                            flierprops=dict(markerfacecolor='r'),
                            medianprops=dict(color='w'),
                            widths=.6)

                        newax.set_xlim(
                            -np.max(evs[evs != 0])-1,
                            -np.min(evs[evs != 0])+1)
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
    """Write into a string the transfers made for a gw

    Args:
        x (array): _description_
        player_names (pd.DataFrame): player names

    Returns:
        str: transfered players
    """
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
    """Write into a list the freehit team

    Args:
        x (array): _description_
        player_names (pd.DataFrame): player names

    Returns:
        list: transfered players
    """
    fh_list = []

    for fh_player in x:
        fh_list.append(
            player_names.loc[fh_player]['first_name'] +
            ' ' +
            player_names.loc[fh_player]['second_name'])

    return fh_list


def imscatter(x, y, image, ax=None, zoom=1):
    """stackoverflow.com/questions/35651932/plotting-img-with-matplotlib/35651933"""
    if ax is None:
        ax = plt.gca()

    try:
        image = plt.imread(image)
    except TypeError:
        pass

    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists