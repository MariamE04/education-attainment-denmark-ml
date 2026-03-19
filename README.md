# Education Attainment in Denmark – BI & Machine Learning Project

## Group members 
- Abbas Mahmoud Badreddine
- Mariam Lumiere El Mir


## Project Overview
This project analyses how demographic factors such as **gender, origin, and age** relate to **educational attainment in Denmark**.  
The analysis is based on data from **Statistics Denmark (DST)** and focuses on individuals aged **20–39 years** between **2020–2024**.

The project follows a Business Intelligence workflow including data preparation, exploratory data analysis, machine learning modelling, and a prototype business application.

---

## Objectives
The goal of the project is to explore whether demographic characteristics influence educational outcomes and whether these variables can be used to **predict educational attainment**.

---

## Research Questions
- **RQ1:** Does gender influence the educational attainment of young adults in Denmark?
- **RQ2:** Does origin (Danish origin, immigrants, descendants) influence the level of education individuals achieve?
- **RQ3:** Can educational attainment be predicted using demographic characteristics such as gender, origin, and age?

---

## Hypotheses
- **H1:** There are significant differences in educational attainment between men and women.
- **H2:** Individuals with Danish origin are more likely to complete higher levels of education compared to immigrants and descendants.
- **H3:** Gender, origin, and age can be used to predict educational attainment.

---

## Dataset
The dataset used in this project comes from **Statistics Denmark (DST)**.  
It contains aggregated statistics about educational attainment by:

- Gender
- Origin (Danish origin, immigrants, descendants)
- Age group
- Education type
- Year (2020–2024)

The data was cleaned and structured using **Python and Pandas**.

---

## Project Structure

The project is divided into four stages:

### Stage 1 – Problem Formulation
Defines the context, research questions, and hypotheses.

### Stage 2 – Data Preparation & Exploration
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical summaries and visualisations

### Stage 3 – Data Modelling
Machine learning models were applied to explore predictive relationships:

- Decision Tree Classifier
- Random Forest Classifier
- K-Means Clustering

Model evaluation was performed using metrics such as:
- Accuracy
- Precision
- Recall
- Confusion Matrix

### Stage 4 – Business Application

The results are presented through a Streamlit web application that allows users to explore the data, visualisations, and model predictions interactively.
The application includes dashboards, visualisations, and a prediction interface designed for non-technical users.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit (prototype application)

---

## Key Findings
The analysis suggests that **age and origin have a stronger influence on educational attainment than gender**.  
However, the predictive models achieved relatively low accuracy, indicating that additional factors beyond demographic variables likely influence educational outcomes.

---

## Running the Application

To run the Streamlit dashboard locally:

1. Clone the repository
2. Install the required packages
3. Run the following command:

streamlit run app.py

The dashboard will open in your browser.

## Potential Applications
The insights from this analysis may support:

- Educational policy discussions
- Research on social inequality
- Data-driven decision making in education systems
