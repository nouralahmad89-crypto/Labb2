# Labb 2: Klassificering av Pichu och Pikachu med KNN
# importera bibliotek och create .venv file
import matplotlib.pyplot as plt
import math
import numpy as np
import re

# läs in data från datapoints.txt fil 
dimansion=[] # en lista för att spara width och height
Labl= [] # en lista för att spara labels
with open("datapoints.txt" ,"r") as f:
    f.readline()  # hoppa över första raden (header)
    for line in f:
        parts= line.strip().split(",")
        width= float (parts[0])
        height= float(parts[1])
        Label= float(parts[2])
        dimansion.append([width , height])
        Labl.append(Label)       
        #print(dimansion)
        #print(Labl)
dimansion = np.array(dimansion) # konvertera till numpy-arrays
Labl = np.array(Labl)
# Plotta data 
plt.figure(figsize=(7,5))
pichu= np.where(Labl==0) # hämta index för pichu.
pikachu = np.where(Labl == 1) # hämta index för pikachu.
plt.scatter(dimansion[pichu, 0], dimansion[pichu, 1],
            color="blue", label="Pichu (0)", s=30, alpha=0.5) # plottar pichu punkter i blått
plt.scatter(dimansion[pikachu, 0], dimansion[pikachu, 1],
            color="yellow", label="Pikachu (1)", s=60, alpha=0.7) # plottar pikachu punkter i gult
plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Pichu vs Pikachu - Data Points")
plt.legend()
plt.grid(True)
plt.show()
# läs in test punnkter
testpoints=[]
with open("testpoints.txt", "r") as test:
    test.readline()
    for T in test:
        T= T.strip()
        point= T.split("(")[1].split(")")[0]
        x,y= map(float, point.split(","))
        testpoints.append([x,y])   
    
# en funktion för att beräkna avståndet mellan punkter
def euclidean(a, b):
     return np.sqrt(np.power(a[0]-b[0], 2) + np.power(a[1]-b[1], 2))
for pt in testpoints: # loopa igenom testpunkterna och klassificera
    try:
        if pt[0]<0 or pt[1]<0:  # Kontrollera negativa värden
            raise ValueError(f"punkten{pt[0]}{pt[1]} inhåller negativa väredn")
        distances = [euclidean(pt, x) for x in dimansion]
        nearest_index = np.argmin(distances)      # index för närmaste punkt
        predicted_label = Labl[nearest_index]     # hämta label för den närmaste
        if predicted_label == 1:
            print(f"Sample with (width, height): ({pt[0]}, {pt[1]}) classified as Pikachu")
        else:
            print(f"Sample with (width, height): ({pt[0]}, {pt[1]}) classified as Pichu")
    except ValueError as err:  
        print(err)
        print(f"Fel: punkten ({pt[0]}, {pt[1]}) innehåller icke-numeriska värden, hoppar över")
        continue
