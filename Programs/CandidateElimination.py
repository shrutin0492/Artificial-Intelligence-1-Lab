#CandidateElimination
import numpy as np
import pandas as pd

data = pd.read_csv('ENJOYSPORT.csv') #make sure to tailor this path if needed!
concepts = np.array(data.iloc[:,0:-1])
print("\nInstances are:\n",concepts)
target = np.array(data.iloc[:,-1])
print("\nTarget Values are: ",target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and genearal_h")
    print("\nSpecific Boundary: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("\nGeneric Boundary: ",general_h)

    for i, h in enumerate(concepts):
        print("\nInstance", i+1 , "is ", h)
        if target[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_h)):
                if h[x]!= specific_h[x]:
                    specific_h[x] ='?'
                    general_h[x][x] ='?'

        if target[i] == "no":
            print("Instance is Negative ")
            for x in range(len(specific_h)):
                if h[x]!= specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Specific Bundary after ", i+1, "Instance is ", specific_h)
        print("Generic Boundary after ", i+1, "Instance is ", general_h)
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Final Specific_h: ", s_final, sep="\n")
print("Final General_h: ", g_final, sep="\n")

"""
#Final Output Only
import csv

a = []

print("\nThe Given Training Data Set")

with open('ENJOYSPORT.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)

num_attributes = len(a[0]) - 1

print("\nThe initial value of hypothesis: ")
S = ['0'] * num_attributes
G = ['?'] * num_attributes

print("\nThe most specific hypothesis S0: [0,0,0,0,0,0]")
print("\nThe most general hypothesis G0: [?,?,?,?,?,?]")

for j in range(0, num_attributes):
    S[j] = a[0][j]

print("\nCandidate Elimination algorithm Hypotheses Version Space Computation\n")
temp = []

for i in range(0, len(a)):
    if a[i][num_attributes] == 'Yes':
        for j in range(0, num_attributes):
            if a[i][j] != S[j]:
                S[j] = '?'

        for j in range(0, num_attributes):
            for k in range(1, len(temp)):
                if temp[k][j] != '?' and temp[k][j] != S[j]:
                    del temp[k]

        print("---------------------------------------------------------")
        print("For Training Example No: {0}, the hypothesis is S{0}".format(i+1), S)

        if len(temp) == 0:
            print("For Training Example No: {0}, the hypothesis is G{0}".format(i+1), G)
        else:
            print("For Positive Training Example No: {0}, the hypothesis is G{0}".format(i+1), temp)

    if a[i][num_attributes] == 'No':
        for j in range(0, num_attributes):
            if S[j] != a[i][j] and S[j] != '?':
                G[j] = S[j]
                temp.append(G)
                G = ['?'] * num_attributes

        print(" ")
        print("For Training Example No: {0}, the hypothesis is S{0}".format(i+1), S)
        print("For Training Example No: {0}, the hypothesis is G{0}".format(i+1), temp)
"""
