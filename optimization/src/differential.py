import streamlit as st

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath

from team_optimization import Team_Optimization


def write():
    st.title('FPL - Differential Model')
    st.header(
        """
        Differential FPL Optimization.
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


        col1, col2, col3 = st.columns(3)
        with col1:
            nb_diff = st.slider("# Differentials", min_value=0, max_value=11, value=3, step=1)
        with col2:
            threshold = st.slider("Threshold", min_value=1, max_value=100, value=10, step=1)
        with col3:
            rank = st.selectbox("Rank", ['Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K', 'Top_500K'], 4)


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

            to.differential_model(
                nb_differentials=nb_diff,
                threshold=threshold,
                target=rank)

            df, chip_strat, total_ev, total_obj = to.solve(
                model_name="differential",
                log=True,
                time_lim=0)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Value", np.round(total_ev, 2))
            with col2:
                st.metric("Objective Function Value", np.round(total_obj, 2))

            fig, ax = plt.subplots(figsize=(16, 12))
            # Set up the axis limits with a bit of padding
            ax.set_ylim(0, 15 + 1)
            ax.set_xlim(0, (horizon+1)*16+2.5)
            ax.axis('off')
            header_pos = 15.25

            color_position = {'G': "#ebff00", 'D': "#00ff87", 'M': "#05f0ff", 'F': "#e90052"}

            for j, row in to.initial_team_df.iterrows():
                rectangle = patches.Rectangle(
                    (0, 14-j),
                    12, .75,
                    facecolor=color_position[row['Pos']])
                ax.add_patch(rectangle)
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width()/2.0
                cy = ry + rectangle.get_height()/2.0
                ax.annotate(
                    'TAA' if row['Name']=='Alexander-Arnold' else row['Name'],
                    (cx, cy),
                    color='black',
                    weight='bold',
                    fontsize=14,
                    ha='center',
                    va='center')

            # Column headers
            ax.text(
                cx, header_pos,
                'Base',
                fontsize=14, weight='bold', ha='center')

            # Bench separator
            ax.plot(
                [0, 12],
                [3.875, 3.875],
                ls=':', lw='2.5', c='grey')

            # Header separator
            ax.plot(
                [0, 12],
                [15., 15.],
                ls='-', lw='2.5', c='grey')

            for i, gw in enumerate(np.sort(df.GW.unique())):
                df_gw = df.loc[df.GW==gw].reset_index(drop=True)

                for j, row in df_gw.iterrows():
                    rectangle = patches.Rectangle(
                        ((i+1)*16, 14-j),
                        12, .75,
                        facecolor=color_position[row['Pos']])
                    ax.add_patch(rectangle)
                    rx, ry = rectangle.get_xy()
                    cx = rx + rectangle.get_width()/2.0
                    cy = ry + rectangle.get_height()/2.0
                    ax.annotate(
                        (
                            ('TAA' if row['Name']=='Alexander-Arnold' else row['Name']) +
                            (" (C)" if row['Cap'] else '') +
                            (" (V)" if row['Vice'] else '') +
                            (" (D)" if row['Ownership'] < threshold else '')),
                        (cx, cy),
                        color='black',
                        weight='bold',
                        fontsize=14,
                        ha='center',
                        va='center')

                # Column headers
                ax.text(
                    cx, header_pos,
                    str(gw),
                    fontsize=14, weight='bold', ha='center')
                if chip_strat[i] is not None:
                    ax.text(
                        cx, header_pos+1,
                        chip_strat[i],
                        fontsize=14, weight='bold', ha='center')

                # Bench separator
                ax.plot(
                    [(i+1)*16, (i+1)*16+12],
                    [3.875, 3.875],
                    ls=':', lw='2.5', c='grey')

                if i == 0:
                    transfers = to.initial_team_df.append(df_gw, ignore_index=True)[['Name', 'Pos']]
                    transfers = transfers.drop_duplicates(keep=False).sort_index()

                    for pos in ['G', 'D', 'M', 'F']:
                        transfer_ = transfers.loc[transfers.Pos==pos]

                        for _ in range(int(transfer_.shape[0]/2)):
                            # Plot the lines
                            ax.add_patch(
                                bezier_path(
                                    (12, 14-transfer_.head(1).index[0]+.75/2),
                                    (16, 14-transfer_.tail(1).index[0]+15+.75/2)))
                            transfer_ = transfer_.drop([
                                transfer_.head(1).index[0],
                                transfer_.tail(1).index[0]])

                else:
                    transfers = df.loc[df.GW==gw-1].append(df_gw, ignore_index=True)[['Name', 'Pos']]
                    transfers = transfers.drop_duplicates(keep=False).sort_index()

                    for pos in ['G', 'D', 'M', 'F']:
                        transfer_ = transfers.loc[transfers.Pos==pos]

                        for _ in range(int(transfer_.shape[0]/2)):
                            # Plot the lines
                            ax.add_patch(
                                bezier_path(
                                    (i*16+12, 14-transfer_.head(1).index[0]+.75/2),
                                    ((i+1)*16, 14-transfer_.tail(1).index[0]+15+.75/2)))
                            transfer_ = transfer_.drop([
                                transfer_.head(1).index[0],
                                transfer_.tail(1).index[0]])

                # Header separator
                ax.plot(
                    [(i+1)*16, (i+1)*16+12],
                    [15., 15.],
                    ls='-', lw='2.5', c='grey')

            st.pyplot(fig, ax)
            plt.close(fig)


def bezier_path(p1, p2, color='white'):
    Path = mpath.Path
    x1, y1 = p1
    x2, y2 = p2

    if y2 != y1:
        path_data = [
            (Path.MOVETO, (x1, y1)),
            (Path.CURVE3, (x1+(x2-x1)/2, y1)),
            (Path.CURVE3, (x1+(x2-x1)/2, y1+(y2-y1)/2)),
            (Path.CURVE3, (x1+(x2-x1)/2, y2)),
            (Path.CURVE3, (x2, y2)),
            ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, ec=color, fc='none', zorder=2, lw=2)

    else:
        path_data = [
            (Path.MOVETO, (x1, y1)),
            (Path.LINETO, (x2, y2)),
            ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, ec=color, fc='none', zorder=2, lw=2)

    return patch