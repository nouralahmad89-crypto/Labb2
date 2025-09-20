# Labb 2: Klassificering av Pichu och Pikachu med KNN
# importera bibliotek och create .venv file
import matplotlib as lp
import math
import pandas as pa
import numpy as np

# Läs in datan och spara i lämplig datastruktur
dataf = pa.read_csv("datapoints.txt", skiprows=1, header=None, names=["width","height","label"])
print(dataf.shape) # double check the dataframe , number of rows and columns
dimansion = dataf[["width", "height"]].values
Labl = dataf["label"].values

# Plotta data

