""" Labb 2: Klassificering av Pichu och Pikachu med KNN
 importera bibliotek och skapa .venv file"""

import matplotlib.pyplot as plt
import numpy as np
import random

# 1) läs in data från datapoints.txt 

dimansion=[] # en lista för att spara width och height
labl= [] # en lista för att spara labels
try:
   with open("datapoints.txt" ,"r") as f:
    f.readline()  # hoppa över första raden (header)
    for line in f:
        parts= line.strip().split(",")
        width= float (parts[0])
        height= float(parts[1])
        label= float(parts[2])
        dimansion.append([width , height])
        labl.append(label)       
        #print(dimansion) check om alla ser ut bra
        #print(Labl)
except FileNotFoundError:
    raise FileNotFoundError ("File not found, make sure path is correct!")        
dimansion = np.array(dimansion) # konvertera till numpy-arrays
labl = np.array(labl)

# ----- Plotta data -----
plt.figure(figsize=(7,5))
pichu= np.where(labl==0) # hämta index för pichu.
pikachu = np.where(labl == 1) # hämta index för pikachu.
plt.scatter(dimansion[pichu, 0], dimansion[pichu, 1],
            color="blue", label="Pichu (0)", s=30, alpha=0.5) # plottar pichu punkter i blått
plt.scatter(dimansion[pikachu, 0], dimansion[pikachu, 1],
            color="yellow", label="Pikachu (1)", s=60, alpha=0.7) # plottar pikachu punkter i gult
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Pichu vs Pikachu - Data Points")
plt.legend()
plt.grid(True)
plt.show()

# läs in test punkter
testpoints=[]
try:
    with open("testpoints.txt", "r") as test:
     test.readline()
     for T in test:
        T= T.strip()
        point= T.split("(")[1].split(")")[0] 
        x,y= point.split(",")
        testpoints.append([x,y])
except FileNotFoundError:   
       raise FileNotFoundError("File not found, make sure path is correct!")
     
# en funktion för att beräkna avståndet mellan punkter
def euclidean(a, b):
     return np.sqrt(np.power(a[0]-b[0], 2) + np.power(a[1]-b[1], 2))

 #Klassificera testpunkter med 1-NN (närmsta granne)
def classify_1nn(testpoints, dimansion, labl):    
    for pt in testpoints: # loopa igenom testpunkterna och klassificera
        try:
           # convert till float
          pt[0]= float(pt[0]) 
          pt[1]= float(pt[1])
        except ValueError :  
            print(f"Fel: punkten ({pt[0]}, {pt[1]}) innehåller icke-numeriska värden, hoppar över")
            continue # Hoppa över den här punkten               
        if pt[0]<0 or pt[1]<0:  # Kontrollera negativa värden
          print(f"punkten{pt[0]},{pt[1]} inhåller negativa väredn, hpoppa över")
          continue # hoppa över den här punkten
    # Beräkna avstånd och klassificera   
        distances = [euclidean(pt, x) for x in dimansion]
        nearest_index = np.argmin(distances)      # index för närmaste punkt
        predicted_label = labl[nearest_index]     # hämta label för den närmaste
        if predicted_label == 1:
          print(f"Sample with (width, height): ({pt[0]}, {pt[1]}) classified as Pikachu")
        else:
          print(f"Sample with (width, height): ({pt[0]}, {pt[1]}) classified as Pichu")
classify_1nn(testpoints, dimansion, labl)   # anropa funktion
"""
Nu sk vi låta användern ange width och height 
och vi anropar funktionen
"""
W= input("input figure width:") 
H= input("iput figure height:")  
user_input=[]
user_input.append([W,H]) 
classify_1nn(user_input, dimansion, labl)

# 10-NN (k=10)
def classify_10knn (testpoints,labl,dimansion,k):
    resluts=[] # spara all resluts from Euclidean methon
    for p in testpoints:
         # Kontrollera negativa eller icke-numeriska värden
        try:
            p[0], p[1] = float(p[0]), float(p[1])
        except ValueError:
            print(f"hoppa över icke-numeriska vären({p[0]}, {p[1]})")
            continue    
        if p[0] < 0 or p[1] < 0:
            print(f" hoppa över negativa väredn: ({x}, {y})")
            continue
        resluts=[euclidean(p,x)  for x in dimansion] # Beräkna alla avstånd
        d = list(zip(resluts, range(len(resluts)))) # (results, index)
        d.sort()
        k_values= d[:k] # Ta de k närmaste indexen
        counter_pichu=0
        counter_pikachu=0
        for value in k_values:
            if labl[value[1]]==0:
                counter_pichu+=1
            elif labl[value[1]]==1:
              counter_pikachu+=1
        if counter_pichu > counter_pikachu:  # Bestäm klass baserat på flest röster
          predicted = 0
        else:
          predicted = 1  
        if predicted==0:
            print(f"Sample with (width, height): {p[0]}, {p[1]} classified as Pichu with KNN-{k}")
        else:
              print(f"Sample with (width, height):{p[0]}, {p[1]} classified as Pikatchu with KNN-{k}")        
classify_10knn (testpoints,labl,dimansion, k=10)

# -----tänkte plota testpunkter -----
testpoints= np.array(testpoints) # convert till numpy-array
plt.figure(figsize=(7,5))
plt.scatter(testpoints[:,0], testpoints[:,1], color='red',
             label='Testpunkter(?)', s=45 , alpha= 0.9) # plotta test punkter i röd
plt.scatter(dimansion[pichu, 0], dimansion[pichu, 1],
            color="blue", label="Pichu (0)", s=30, alpha=0.5) # plottar pichu punkter i blått
plt.scatter(dimansion[pikachu, 0], dimansion[pikachu, 1],
            color="yellow", label="Pikachu (1)", s=60, alpha=0.7) # plottar pikachu punkter i gult
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Pichu vs Pikachu(Med Test Punkter) - Data Points")
plt.legend()
plt.grid(True)
plt.show()

""" BONUSUPPGIFTER
 3)Dela in ursprungsdatan slumpmässigt """
# Extrahera punkterna med hjälp av index (pichu) och (pikachu)
pichu_points = dimansion[pichu] 
pikachu_points = dimansion[pikachu]
#Slumpa ordningen i varje klass
np.random.shuffle(pichu_points) 
np.random.shuffle(pikachu_points)
 # dela test/träning punkter(class pichu) 
pichu_train, pichu_test = pichu_points[:50], pichu_points[50:75]
 # dela test/träning punkter(class pikachu)
pikachu_train, pikachu_test = pikachu_points[:50], pikachu_points[50:75]
X_train = np.vstack((pichu_train, pikachu_train)) # slår ihop pichu och pikachu till en träningsset.
Y_train = np.array([0]*len(pichu_train) + [1]*len(pikachu_train)) 
X_test = np.vstack((pichu_test, pikachu_test)) #slår ihop pichu och pikachu till en test set.
Y_test = np.array([0]*len(pichu_test) + [1]*len(pikachu_test)) 
# slumpmäsig blandning av train och test data 
train_indx= np.arange(len(X_train)) # array innehåller alla indecies
np.random.shuffle(train_indx) # shuffle med index så punkter och label fortfarande matchar
X_train= X_train[train_indx]
Y_train= Y_train[train_indx]
test_indx = np.arange(len(X_test))
np.random.shuffle(test_indx) # blander punkter
X_test = X_test[test_indx]
Y_test = Y_test[test_indx]
 #100 points för träning data
print("X_train shape:", X_train.shape) 
print("y_train shape:", Y_train.shape)
# 50 points för test data
print("X_test shape:", X_test.shape)
print("y_test shape:", Y_test.shape)

""" 4)  Beräkna noggranheten
  accuracy = (#TP+#TN)/(total)
"""
def evaluate_KNN(X_train, Y_train, X_test, Y_test, k=10):
    """
    Klassificera alla testpunkter med K-10NN och beräkna accuracy.
    Pikachu = positiv, Pichu = negativ
    """
    counter = 0  # räknare för korrekta klassificeringar
    for i, pt in enumerate(X_test):  # loop genom alla testpunkter
        distances = [euclidean(pt, x) for x in X_train] #beräkna avstånd till alla träningspunkter
        k_indices = np.argsort(distances)[:k] # hämta index för de k närmaste punkterna
        k_labels = Y_train[k_indices]  #hämta deras labels
        count_0 = sum(1 for lbl in k_labels if lbl == 0)   #  majoritetsröstning
        count_1 = sum(1 for lbl in k_labels if lbl == 1)
        if count_0 > count_1:
            predicted = 0
        else:
            predicted = 1
        if predicted == Y_test[i]: # jämför med facit
            counter += 1
    accuracy = counter / len(Y_test)  #beräkna accuracy 
    return accuracy

# Nu ska vi kör 10 gånger
accuracies = []  # här sparas alla 10 accuracy

for run in range(10):
    # slumpa om ordningen för pichu och pikachu
    np.random.shuffle(pichu_points)
    np.random.shuffle(pikachu_points)
    # dela train/test (samma som innan)
    pichu_train, pichu_test = pichu_points[:50], pichu_points[50:75]
    pikachu_train, pikachu_test = pikachu_points[:50], pikachu_points[50:75]
    X_train = np.vstack((pichu_train, pikachu_train))
    Y_train = np.array([0]*len(pichu_train) + [1]*len(pikachu_train))
    X_test = np.vstack((pichu_test, pikachu_test))
    Y_test = np.array([0]*len(pichu_test) + [1]*len(pikachu_test))
    # blanda train/test så ordningen blir slumpmässig
    train_idx = np.arange(len(X_train))
    np.random.shuffle(train_idx)
    X_train, Y_train = X_train[train_idx], Y_train[train_idx]

    test_idx = np.arange(len(X_test))
    np.random.shuffle(test_idx)
    X_test, Y_test = X_test[test_idx], Y_test[test_idx]
    # beräkna accuracy
    acc = evaluate_KNN(X_train, Y_train, X_test, Y_test, k=10)
    accuracies.append(acc)
    print(f"Körning {run+1}: Accuracy = {acc:.2f}")

# --- Plotta ---
plt.plot(range(1, 11), accuracies, marker="o", color="blue")
plt.title("Accuracy för 10 upprepade körningar (10-NN)")
plt.xlabel("Körning")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# --- Medelaccuracy ---
mean_acc = np.mean(accuracies)
print(f"Medelaccuracy över 10 körningar: {mean_acc:.2f}")

 


        
       

       

      

   


