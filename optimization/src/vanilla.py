import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from highlight_text import fig_text
from matplotlib.colors import ListedColormap

from team_planner import Team_Planner


def write():
    st.title('FPL - Vanilla Model')
    st.header(
        """
        Vanilla FPL Optimization.
        """)

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


    if st.button('Run Optimization'):

        with st.spinner("Running Optimization ..."):
            tp = Team_Planner(
                team_id=35868,
                horizon=horizon,
                noise=False,
                premium=True if premium=='Premium' else False)

            tp.build_model(
                model_name="vanilla",
                objective_type='decay' if decay != 0 else 'linear',
                decay_gameweek=decay,
                vicecap_decay=vicecap_decay,
                decay_bench=[gk_weight, first_bench_weight, second_bench_weight, third_bench_weight],
                ft_val=ft_val,
                itb_val=itb_val,
                hit_val=hit_val)

            df = tp.solve(
                model_name="vanilla",
                log=False,
                time_lim=0)

            fig, ax = plt.subplots(figsize=(8, 6))
            # Set up the axis limits with a bit of padding
            ax.set_ylim(0, 15 + 1)
            ax.set_xlim(0, (horizon+1)*15 + 2.5)
            ax.axis('off')
            header_pos = 15.25

            # st.dataframe(tp.initial_team_df)
            for j, row in tp.initial_team_df.iterrows():
                rectangle = patches.Rectangle(
                    (0, 14-j),
                    12, .75,
                    facecolor={'G': "#ebff00", 'D': "#00ff87", 'M': "#05f0ff", 'F': "#e90052"}[row['Pos']])
                ax.add_patch(rectangle)
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width()/2.0
                cy = ry + rectangle.get_height()/2.0
                ax.annotate(
                    'TAA' if row['Name']=='Alexander-Arnold' else row['Name'],
                    (cx, cy),
                    color='black',
                    weight='bold',
                    fontsize=9,
                    ha='center',
                    va='center')

            # Column headers
            ax.text(
                cx, header_pos,
                'Base',
                weight='bold', ha='center',
                color={'FH': "#ebff00", 'BB': "#00ff87", 'WC': "#05f0ff", 'TC': "#e90052", None: 'black'}[None])

            # Bench separator
            ax.plot(
                [0, 12],
                [3.875, 3.875],
                ls=':', lw='1.5', c='grey')

            for i, gw in enumerate(np.sort(df.GW.unique())):
                df_gw = df.loc[df.GW == gw].reset_index(drop=True)
                # st.dataframe(df_gw)

                for j, row in df_gw.iterrows():
                    rectangle = patches.Rectangle(
                        ((i+1)*16, 14-j),
                        12, .75,
                        facecolor={'G': "#ebff00", 'D': "#00ff87", 'M': "#05f0ff", 'F': "#e90052"}[row['Pos']])
                    ax.add_patch(rectangle)
                    rx, ry = rectangle.get_xy()
                    cx = rx + rectangle.get_width()/2.0
                    cy = ry + rectangle.get_height()/2.0
                    ax.annotate(
                        'TAA' if row['Name']=='Alexander-Arnold' else row['Name'],
                        (cx, cy),
                        color='black',
                        weight='bold',
                        fontsize=9,
                        ha='center',
                        va='center')

                # Column headers
                ax.text(
                    cx, header_pos,
                    str(gw),
                    weight='bold', ha='center',
                    color={'FH': "#ebff00", 'BB': "#00ff87", 'WC': "#05f0ff", 'TC': "#e90052", None: 'black'}[None])

                # Bench separator
                ax.plot(
                    [(i+1)*16, (i+1)*16+12],
                    [3.875, 3.875],
                    ls=':', lw='1.5', c='grey')

            st.pyplot(fig, ax)
            plt.close(fig)
