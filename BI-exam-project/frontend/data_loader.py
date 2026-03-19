import pandas as pd
import streamlit as st


@st.cache_data
def load_main_data():
    df = pd.read_csv("../data/education_clean.csv")

    year_cols = ["2020", "2021", "2022", "2023", "2024"]
    for col in year_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


@st.cache_data
def load_parents_data():
    parents_df = pd.read_csv("../data/Parent-education_clean.xlsx")

    if "total_2020_2024" in parents_df.columns:
        parents_df["total_2020_2024"] = pd.to_numeric(
            parents_df["total_2020_2024"], errors="coerce"
        ).fillna(0)

    year_cols = ["2020", "2021", "2022", "2023", "2024"]
    for col in year_cols:
        if col in parents_df.columns:
            parents_df[col] = pd.to_numeric(
                parents_df[col], errors="coerce"
            ).fillna(0)

    return parents_df