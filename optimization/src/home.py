import streamlit as st


def write():
    st.title('FPL - Home')
    with st.spinner("Loading About ..."):
        st.header(
            """
            FPL Optimization.
            """
            )
        st.write(
            """
            Solve Optimal team for FPL.
            """)