import streamlit as st
from team_planner import Team_Planner

def write():
    st.title('FPL - Vanilla Model')
    st.header(
        """
        Vanilla FPL Optimization.
        """)

    st.subheader("Optimization parameters")

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
    
    
    if st.button(
            'Run Optimization'):

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

            st.dataframe(df)