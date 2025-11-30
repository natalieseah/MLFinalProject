from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np

df = pd.read_csv("MIC.csv")

target_col = "Obesity_Level"  # TODO: update (not this anymore)
predictor_cols = [c for c in df.columns if c != target_col]

# ----- UI LAYOUT -----
app_ui = ui.page_navbar(

    # README TAB
    ui.nav(
        "README",
        ui.h2("Project Overview: Obesity Level Preditions from Nutritional and Physical Characteristics"),
        ui.markdown("""
        ## Dataset
        Using **obesity_cleaned.csv**, this project explores which factors 
        predict obesity category.

        ## Goals
        - Compare predictive models  
        - Provide an interactive dashboard  
        - Allow users to input predictor values and get predictions  

        ## Models Included
        - **k-Nearest Neighbors (KNN)**  
        - **k-Means Clustering** (unsupervised – provides cluster membership)  
        - **Multinomial Logistic Regression**  
        - **Multiple Linear Regression**  
        """)
    ),

    # DATA TAB ---------------------------------------------------
    ui.nav(
        "Data",
        ui.h3("Raw Dataset"),
        ui.output_data_frame("table")
    ),

    # MODEL TAB --------------------------------------------------
    ui.nav(
        "Modeling",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Choose a Model"),
                ui.input_select(
                    "model_choice",
                    "Model Type:",
                    {
                        "knn": "K-Nearest Neighbors",
                        "kmeans": "K-Means Clustering",
                        "mlogit": "Multinomial Logistic Regression",
                        "linreg": "Multiple Linear Regression"
                    }
                ),

                ui.input_slider("split", "Train/Test Split (%)",
                                min=50, max=90, value=80),

                ui.h4("Predictor Inputs (for supervised models)"),
                *[
                    ui.input_numeric(f"pred_{col}", col, float(df[col].mean()))
                    for col in predictor_cols
                ],

                ui.h4("K-Means Settings"),
                ui.input_numeric("k_clusters", "Number of Clusters", 3),

                ui.hr(),
                ui.input_action_button("run_model", "Run Model",
                                       class_="btn-primary")
            ),

            ui.card(
                ui.h3("Model Output"),
                ui.output_text("result")
            )
        )
    )
)