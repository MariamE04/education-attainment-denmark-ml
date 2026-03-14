import streamlit as st
import pandas as pd
import altair as alt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Education Analysis Denmark",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Education Level Analysis in Denmark")

st.markdown(
"""
This interactive dashboard presents the distribution of education levels in Denmark.
Users can explore demographic patterns and visualise how education attainment varies
across gender, age groups, and origin.
"""
)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/education_clean.csv")

years = ["2020","2021","2022","2023","2024"]

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("Filters")

year = st.sidebar.selectbox(
    "Select year",
    years
)

gender = st.sidebar.selectbox(
    "Select gender",
    df["køn"].unique()
)

age = st.sidebar.selectbox(
    "Select age group",
    df["alder"].unique()
)

origin = st.sidebar.selectbox(
    "Select origin",
    df["herkomst"].unique()
)

# -------------------------------
# APPLY FILTERS
# -------------------------------
filtered_df = df[
    (df["køn"] == gender) &
    (df["alder"] == age) &
    (df["herkomst"] == origin)
]

# -------------------------------
# METRICS
# -------------------------------
population = int(filtered_df[year].sum())
st.metric("Population in selected group", population)

# -------------------------------
# DASHBOARD LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# DATASET PREVIEW
# -------------------------------
with col1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df)

# -------------------------------
# EDUCATION LEVEL DISTRIBUTION
# -------------------------------
with col2:

    st.subheader("Education Level Distribution")

    chart_data = (
        filtered_df.groupby("education_level")[year]
        .sum()
        .reset_index()
    )

    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("education_level", title="Education Level"),
        y=alt.Y(year, title="Population"),
        color="education_level",
        tooltip=["education_level", year]
    ).properties(
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# TREND GRAPH
# -------------------------------
st.subheader("Education Trends Over Time")

trend_data = (
    filtered_df.groupby("education_level")[years]
    .sum()
    .reset_index()
    .melt(id_vars="education_level", var_name="year", value_name="population")
)

trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
    x="year",
    y="population",
    color="education_level",
    tooltip=["education_level","year","population"]
)

st.altair_chart(trend_chart, use_container_width=True)

# -------------------------------
# INTERPRETATION
# -------------------------------
st.subheader("Interpretation")

st.markdown(
"""
The dashboard visualises how education levels are distributed across demographic groups.

The results show that education attainment is unevenly distributed across the population.
Some education levels represent significantly larger groups than others.

By using the interactive filters, users can explore how education patterns vary across
gender, age groups, and origin.

This prototype demonstrates how demographic data can be transformed into an
interactive decision-support tool for policymakers, researchers, and educational institutions.
"""
)