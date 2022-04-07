import streamlit as st
import awesome_streamlit as ast

from src import (
    home,
    vanilla,
    differential,
    wildcard,
    select_chips,
    automated_chips,
    biased,
    all_in_one,
    sensitivity_analysis)

st.set_page_config(
        page_title="FPL Optimization",
        page_icon="chart_with_upwards_trend",
    )
ast.core.services.other.set_logging_format()

# List of pages available for display
PAGES = {
    "Home": home,
    "Vanilla": vanilla,
    "Differential": differential,
    "Biased": biased,
    "Wildcard": wildcard,
    "Select Chips": select_chips,
    "Automated Chips": automated_chips,
    "Sensitivity Analysis": sensitivity_analysis,
    "All In One": all_in_one,
    }


def main():
    """Core of the app - switches between 'tabs' thanks to the sidebar"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Visit", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)


if __name__ == "__main__":
    main()
