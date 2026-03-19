import streamlit as st
import pandas as pd
import altair as alt

from data_loader import load_main_data, load_parents_data
from models import train_main_models, train_parents_models, predict_main


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Education Analysis Denmark",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------
df = load_main_data()
parents_df = load_parents_data()

years = ["2020", "2021", "2022", "2023", "2024"]

# -------------------------------
# TRAIN MODELS
# -------------------------------
main_models = train_main_models(df)
parents_models = train_parents_models(parents_df)

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose page",
    [
        "Dashboard",
        "Prediction - Demographics",
    ]
)

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("📊 Education Level Analysis in Denmark")

    st.markdown("""
    This dashboard explores how gender, age, and origin relate to education level in Denmark.
    Social background is included as a supporting analysis.
    """)

    st.sidebar.header("Dashboard Filters")

    year = st.sidebar.selectbox("Select year", years)

    gender = st.sidebar.selectbox(
        "Select gender",
        sorted(df["køn"].dropna().unique())
    )

    age = st.sidebar.selectbox(
        "Select age group",
        sorted(df["alder"].dropna().unique())
    )

    origin = st.sidebar.selectbox(
        "Select origin",
        sorted(df["herkomst"].dropna().unique())
    )

    selected_parent = st.sidebar.selectbox(
        "Select parents' education",
        sorted(parents_df["parent_education"].dropna().unique())
    )

    filtered_df = df[
        (df["køn"] == gender) &
        (df["alder"] == age) &
        (df["herkomst"] == origin)
    ]

    parent_filtered_df = parents_df[
        parents_df["parent_education"] == selected_parent
    ]

    population = int(filtered_df[year].sum()) if not filtered_df.empty else 0
    parent_total = (
        int(parent_filtered_df["total_2020_2024"].sum())
        if not parent_filtered_df.empty else 0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Population in selected demographic group", population)
    m2.metric("Observed count in selected parent group", parent_total)
    m3.metric("Rows in filtered demographic data", len(filtered_df))

    st.subheader("Education Level Distribution")

    chart_data = (
        filtered_df.groupby("education_level")[year]
        .sum()
        .reset_index()
        .sort_values(year, ascending=False)
    )

    if not chart_data.empty:
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("education_level:N", title="Education Level", sort="-y"),
            y=alt.Y(f"{year}:Q", title="Population"),
            color=alt.Color("education_level:N", title="Education Level"),
            tooltip=["education_level", year]
        ).properties(height=380)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No demographic data found for the selected filters.")

    st.subheader("Education Trends Over Time")

    trend_data = (
        filtered_df.groupby("education_level")[years]
        .sum()
        .reset_index()
        .melt(id_vars="education_level", var_name="year", value_name="population")
    )

    if not trend_data.empty:
        trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
            x=alt.X("year:N", title="Year"),
            y=alt.Y("population:Q", title="Population"),
            color=alt.Color("education_level:N", title="Education Level"),
            tooltip=["education_level", "year", "population"]
        ).properties(height=400)

        st.altair_chart(trend_chart, use_container_width=True)
    else:
        st.warning("No trend data available for the selected filters.")

    st.divider()
    st.header("Social Background")

    st.subheader("Observed Outcomes for Selected Parent Group")

    social_grouped = (
        parent_filtered_df.groupby("education")["total_2020_2024"]
        .sum()
        .reset_index()
        .sort_values("total_2020_2024", ascending=False)
    )

    if not social_grouped.empty:
        social_chart = alt.Chart(social_grouped).mark_bar().encode(
            x=alt.X("education:N", title="Youth Education", sort="-y"),
            y=alt.Y("total_2020_2024:Q", title="Count"),
            color=alt.Color("education:N", title="Youth Education"),
            tooltip=["education", "total_2020_2024"]
        ).properties(height=380)

        st.altair_chart(social_chart, use_container_width=True)
    else:
        st.warning("No social background data found for the selected parent group.")

    st.subheader("Youth Education Distribution by Parents' Education")

    social_counts = (
        parents_df.groupby(["parent_education", "education"])["total_2020_2024"]
        .sum()
        .reset_index()
    )

    if not social_counts.empty:
        share_chart = alt.Chart(social_counts).mark_bar().encode(
            x=alt.X("parent_education:N", title="Parents' Education"),
            y=alt.Y(
                "total_2020_2024:Q",
                stack="normalize",
                title="Share of Youth Education"
            ),
            color=alt.Color("education:N", title="Youth Education"),
            tooltip=[
                "parent_education",
                "education",
                alt.Tooltip("total_2020_2024:Q", title="Count")
            ]
        ).properties(height=420)

        st.altair_chart(share_chart, use_container_width=True)
    else:
        st.warning("Could not build the social background distribution chart.")

    st.divider()
    st.header("Model Performance Overview")
    st.caption("Note: Accuracy is low because the data is aggregated and the number of features is limited.")

    perf_df = pd.DataFrame({
        "Model": [
            "Decision Tree - Demographics",
            "Random Forest - Demographics",
            "Decision Tree - Social Background",
            "Random Forest - Social Background"
        ],
        "Accuracy": [
            main_models["dt_acc"],
            main_models["rf_acc"],
            parents_models["dt_acc"],
            parents_models["rf_acc"]
        ]
    })

    st.dataframe(perf_df, use_container_width=True)

    perf_chart = alt.Chart(perf_df).mark_bar().encode(
        x=alt.X("Model:N", sort="-y"),
        y=alt.Y("Accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", title="Model"),
        tooltip=[alt.Tooltip("Accuracy:Q", title="Accuracy", format=".2%")]
    ).properties(height=380)

    st.altair_chart(perf_chart, use_container_width=True)

    st.subheader("Interpretation")

    st.markdown(f"""
    The dashboard visualises observed education patterns across demographic groups
    and parental education groups using real aggregated values from the datasets.

    In this implementation, the demographic model performs better than the social
    background model. The demographic model reached approximately **{main_models["dt_acc"]:.2%}**
    for Decision Tree and **{main_models["rf_acc"]:.2%}** for Random Forest.

    The social background model reached approximately **{parents_models["dt_acc"]:.2%}**
    for Decision Tree and **{parents_models["rf_acc"]:.2%}** for Random Forest.

    However, both models have limited predictive power because the data is aggregated
    and the number of predictive features is small. For this reason, social background
    is mainly used here as a complementary analytical perspective.
    """)

# =========================================================
# PREDICTION - DEMOGRAPHICS
# =========================================================
elif page == "Prediction - Demographics":
    st.title("🎯 Prediction - Demographic Model")

    st.markdown("""
    This page uses the main model trained on:
    - gender
    - age
    - origin

    Only real observed demographic profiles can be selected.
    """)

    st.info("This prediction is based on a simple model and should be interpreted with caution.")
    st.caption("Observed data is more reliable than model predictions.")

    st.write(f"Decision Tree Accuracy: {main_models['dt_acc']:.2%}")
    st.write(f"Random Forest Accuracy: {main_models['rf_acc']:.2%}")

    profiles = main_models["real_profiles"].copy()
    profiles["label"] = (
        profiles["køn"] + " | " +
        profiles["alder"] + " | " +
        profiles["herkomst"]
    )

    selected_label = st.selectbox(
        "Choose observed demographic profile",
        profiles["label"].tolist()
    )

    selected_row = profiles[profiles["label"] == selected_label].iloc[0]

    pred_gender = selected_row["køn"]
    pred_age = selected_row["alder"]
    pred_origin = selected_row["herkomst"]

    if st.button("Predict from demographics"):
        preds = predict_main(main_models, pred_gender, pred_age, pred_origin)

        st.subheader("Predicted education outcome")
        st.write("Decision Tree:", preds["decision_tree"])
        st.write("Random Forest:", preds["random_forest"])

        result_df = df[
            (df["køn"] == pred_gender) &
            (df["alder"] == pred_age) &
            (df["herkomst"] == pred_origin)
        ]

        grouped = (
            result_df.groupby("education_level")["2024"]
            .sum()
            .reset_index()
            .sort_values("2024", ascending=False)
        )

        if not grouped.empty:
            st.subheader("Observed outcomes for this demographic profile")
            st.dataframe(grouped, use_container_width=True)

            result_chart = alt.Chart(grouped).mark_bar().encode(
                x=alt.X("education_level:N", sort="-y", title="Education Level"),
                y=alt.Y("2024:Q", title="Population in 2024"),
                color=alt.Color("education_level:N", title="Education Level"),
                tooltip=["education_level", "2024"]
            ).properties(height=400)

            st.altair_chart(result_chart, use_container_width=True)
        else:
            st.warning("No observed data found for this profile.")