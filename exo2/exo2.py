import math
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import pandas as pd
from graphviz import Graph

dataframe = pd.read_csv("dataset.csv")
nb_people = len(dataframe.index)
print(dataframe.info())

print(dataframe.columns)

print(dataframe.head(10))

def compute_dissimilarity(id1, id2):
    entity_1_age = dataframe.loc[id1][1]
    entity_2_age = dataframe.loc[id2][1]

    entity_1_height = dataframe.loc[id1][2]
    entity_2_height = dataframe.loc[id2][2]

    entity_1_job = dataframe.loc[id1][3]
    entity_2_job = dataframe.loc[id2][3]

    entity_1_city = dataframe.loc[id1][4]
    entity_2_city = dataframe.loc[id2][4]

    entity_1_music_style = dataframe.loc[id1][5]
    entity_2_music_style = dataframe.loc[id2][5]

    if entity_1_job == entity_2_job:
        dissimilarity_job = 0
    else:
        dissimilarity_job = 10
    
    if entity_1_city == entity_2_city:
        dissimilarity_city = 0
    else:
        dissimilarity_city = 15

    if entity_1_music_style == entity_2_music_style:
        dissimilarity_music_style = 0
    elif entity_1_music_style == "technical death metal" and entity_2_music_style == "metal":
        dissimilarity_music_style = 5 
    else:
        dissimilarity_music_style= 15
    
    dissimilarity = math.sqrt(
        (entity_1_age - entity_2_age) ** 2
        + (entity_1_height - entity_2_height)
        + dissimilarity_job
        + dissimilarity_city
        + dissimilarity_music_style
    )

    print("==========")
    entity_1_id = dataframe.loc[id1][0]
    entity_2_id = dataframe.loc[id2][0]
    print(
        f"entity1 {entity_1_id}, entity_2_id {entity_2_id}, dissimilarity: {dissimilarity}"
    )
    return dissimilarity

#build dissimilarity matrix
dm = np.zeros((nb_people, nb_people))
print("compute dissimilarities...")
for entity_1_id in range(nb_people):
    for entity_2_id in range(nb_people):
        dissimilarity = compute_dissimilarity(entity_1_id, entity_2_id)
        dm[entity_1_id, entity_2_id] = dissimilarity

mean_dissimilarity = np.mean(dm)
std_dissimilarity = np.std(dm)

print("====== dissimilarity matrix ======")
print(dm)

print("Mean Dissimilarity:", mean_dissimilarity)
print("Standard Deviation of Dissimilarity:", std_dissimilarity)

print("creating matrix file...")
np.save("dissimilarity_matrix.npy", dm)
print ("matrix file created")