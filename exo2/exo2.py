import math
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import pandas as pd
from graphviz import Graph

dataframe = pd.read_csv("dataset.csv")
nb_people = len(dataframe.index)

def print_data_info():
        # general info on the dataframe
    print("---\ngeneral info on the dataframe")
    print(dataframe.info())

    # print the columns of the dataframe
    print("---\ncolumns of the dataset")
    print(dataframe.columns)

    # print the first 10 lines of the dataframe
    print("---\nfirst lines")
    print(dataframe.head(10))

    # print the correlation matrix of the dataset
    print("---\nCorrelation matrix")
    print(dataframe.corr())

    # print the standard deviation
    print("---\nStandard deviation")
    print(dataframe.std())

def print_specific_value(id, dataframe):
    print("---\nall info on player " + str(id))
    print(dataframe.loc[id])

def main():
    print_data_info(dataframe)
    dissimilarity_matrix()
    render_graph()

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
        dissimilarity_job = 15
    
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
        + 3 * (entity_1_height - entity_2_height) ** 2
        + dissimilarity_job
        + dissimilarity_city
        + dissimilarity_music_style
    )

    print("==========")
    entity_1_id = dataframe.loc[entity_1_id][0]
    entity_2_id = dataframe.loc[entity_2_id][0]
    print(
        f"entity1 {entity_1_id}, entity_2_id {entity_2_id}, dissimilarity: {dissimilarity}"
    )
    return dissimilarity

def dissimilarity_matrix():
    dm = np.zeros((nb_people, nb_people))
    print("compute dissimilarities...")
    for entity_1_id in range(nb_people):
        for entity_2_id in range(nb_people):
            dissimilarity = compute_dissimilarity(entity_1_id, entity_2_id)
            dissimilarity_matrix[entity_1_id, entity_2_id] = dissimilarity
    print("====== dissimilarity matrix ======")
    print(dissimilarity_matrix)

    return dissimilarity_matrix

def render_graph():
    threshold = 15

    dot = Graph(comment="Graph created from complex data", strict=True)
    for entity in range(nb_people):
        entity_id = dataframe.loc[entity_id][0]
        dot.node(entity_id)

    for entity_1 in range(nb_people):
        # we use an undirected graph so we do not need
        # to take the potential reciprocal edge
        # into account
        for entity_2 in range(nb_people):
            # no self loops
            if not entity_1 == entity_2:
                entity_1_id = dataframe.loc[entity_1_id][0]
                entity_2_id = dataframe.loc[entity_2_id][0]
                # use the threshold condition
                # EDIT THIS LINE
                if dissimilarity_matrix[entity_1, entity_2] > threshold:
                    dot.edge(
                        entity_1_id,
                        entity_2_id,
                        color="darkolivegreen4",
                        penwidth="1.1",
                    )
    # visualize the graph
    dot.attr(label=f"threshold {threshold}", fontsize="20")
    graph_name = f"images/complex_data_threshold_{threshold}"
    dot.render(graph_name)

if __name__ == "__main__":
    main()