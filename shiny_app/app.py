from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np

df = pd.read_csv("obesity.csv")

target_col = "Obesity_Level"
predictor_cols = [c for c in df.columns if c != target_col]

# ----- UI LAYOUT -----
app_ui = ui.page_navbar(

    # README TAB
    ui.nav(  # TODO: update data source, predictor variables, and ML pipeline included as needed
        "README",
        ui.h2("Project Overview: Obesity Level Preditions from Nutritional and Physical Characteristics"),
        ui.markdown("""
        ## Data Source
         
        
        ## Target Variable
        Obesity level

        ## Predictor Variables
        -

        ## Goals
        - Determine which nutritional, behavioral, and physical characteristics affect obesity level most
        - Produce models to predict obesity based on these characteristics

        ## Maching Learning Pipeline Implemented By This Aoo
        1. Goal of Prediction: Predict obesity level (categorical)
        2. Model Selection: Multiple Regression, Random Forest, Multinomial Logistic Regression, K-Nearest Neighbors
        3. Loss Function Selection: Standard classification loss for each model
        4. Model Training: Fit models on train split
        5. Model Evaluation: 
        6. Model Refinement: Adjust thresholds accordingly
        
        ## App Usage
        1. See the Data preview tab
        2. Choose a model from the Model tab
        3. Adjust train/test split and other required thresholds
        4. Click 'Run Model' to train and evaluate a prediction
        5. Enter new values for predictors to predict probability using selected model
        """)
    ),

    # DATA TAB
    ui.nav(
        "Data",
        ui.h3("Raw Dataset"),
        ui.output_data_frame("table")
    ),

    # MODEL TAB
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