import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------
# DEMOGRAPHIC MODEL
# Features: køn, alder, herkomst
# Target: education_level
# ---------------------------------------------------
@st.cache_resource
def train_main_models(df):
    data = df.copy()
    data = data[["køn", "alder", "herkomst", "education_level"]].dropna().copy()

    le_gender = LabelEncoder()
    le_age = LabelEncoder()
    le_origin = LabelEncoder()
    le_target = LabelEncoder()

    data["køn_enc"] = le_gender.fit_transform(data["køn"])
    data["alder_enc"] = le_age.fit_transform(data["alder"])
    data["herkomst_enc"] = le_origin.fit_transform(data["herkomst"])
    data["target_enc"] = le_target.fit_transform(data["education_level"])

    X = data[["køn_enc", "alder_enc", "herkomst_enc"]]
    y = data["target_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    real_profiles = (
        data[["køn", "alder", "herkomst"]]
        .drop_duplicates()
        .sort_values(["køn", "alder", "herkomst"])
        .reset_index(drop=True)
    )

    return {
        "dt_model": dt_model,
        "rf_model": rf_model,
        "dt_acc": dt_acc,
        "rf_acc": rf_acc,
        "encoders": {
            "gender": le_gender,
            "age": le_age,
            "origin": le_origin,
            "target": le_target
        },
        "real_profiles": real_profiles
    }


def predict_main(model_package, gender, age, origin):
    le_gender = model_package["encoders"]["gender"]
    le_age = model_package["encoders"]["age"]
    le_origin = model_package["encoders"]["origin"]
    le_target = model_package["encoders"]["target"]

    X_new = pd.DataFrame([{
        "køn_enc": le_gender.transform([gender])[0],
        "alder_enc": le_age.transform([age])[0],
        "herkomst_enc": le_origin.transform([origin])[0]
    }])

    dt_pred = model_package["dt_model"].predict(X_new)[0]
    rf_pred = model_package["rf_model"].predict(X_new)[0]

    return {
        "decision_tree": le_target.inverse_transform([dt_pred])[0],
        "random_forest": le_target.inverse_transform([rf_pred])[0]
    }


# ---------------------------------------------------
# SOCIAL BACKGROUND MODEL
# Feature: parent_education
# Target: education
# Used only for model comparison on dashboard
# ---------------------------------------------------
@st.cache_resource
def train_parents_models(parents_df):
    data = parents_df.copy()
    data = data[["parent_education", "education"]].dropna().copy()

    le_parent = LabelEncoder()
    le_target = LabelEncoder()

    data["parent_enc"] = le_parent.fit_transform(data["parent_education"])
    data["target_enc"] = le_target.fit_transform(data["education"])

    X = data[["parent_enc"]]
    y = data["target_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    return {
        "dt_acc": dt_acc,
        "rf_acc": rf_acc
    }