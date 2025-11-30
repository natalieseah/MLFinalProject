from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np

df = pd.read_csv("obesity_cleaned.csv")

target_col = "Obesity_Level"
predictor_cols = [c for c in df.columns if c != target_col]

