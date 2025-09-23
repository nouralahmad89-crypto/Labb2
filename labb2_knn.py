# Labb 2: Klassificering av Pichu och Pikachu med KNN
# importera bibliotek och create .venv file
import matplotlib.pyplot as plt
import math
import pandas as pa
import numpy as np

# Läs in datan och spara i lämplig datastruktur
dataf = pa.read_csv("datapoints.txt", skiprows=1, header=None, names=["width","height","label"])
print(dataf) # double check the dataframe , number of rows and columns
dimansion = dataf[["width", "height"]].values # alla width och height kommer spara i dimsnsion dataform
Labl = dataf["label"].values # alla labels (0,1) kommer spara i dataform labl

# Plotta data
plt.figure(figsize=(7,5)) # skapa en ny figur
pichu = dataf.query("label==0") # filtrerar data att få pichu punkter
pikachu= dataf.query("label==1") # filtrerar data att få pikachu punkter
# plottar pichu punkter i blått
plt.scatter(pichu["width"], pichu["height"], color="blue", label="Pichu (0)", alpha=0.7)
# plottar pikachu punkter i gul
plt.scatter(pikachu["width"], pikachu["height"], color="Yellow", label="Pikachu (1)", alpha=0.7)
plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Pichu vs Pikachu - Data Points")
plt.legend()
plt.grid(True)
plt.show()
