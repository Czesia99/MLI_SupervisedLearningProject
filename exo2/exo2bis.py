import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

numerical_cols = ['age', 'height']
categorical_cols = ['job', 'city', 'favorite music style']
features = numerical_cols + categorical_cols

#weights
numerical_weights = [0.5, 0.2]
categorical_weights = [0.5, 0.5, 0.1]

num_samples = len(df)
dm = np.zeros((num_samples, num_samples))

for i in range(num_samples):
    for j in range(i + 1, num_samples):
        dissimilarity = 0

        for k, col in enumerate(numerical_cols):
            range_min = min(df[col])
            range_max = max(df[col])
            normalized_diff = abs(df.loc[i, col] - df.loc[j, col]) / (range_max - range_min)
            dissimilarity += numerical_weights[k] * normalized_diff


        for k, col in enumerate(categorical_cols):
            dissimilarity += categorical_weights[k] * (df.loc[i, col] != df.loc[j, col])

        dm[i, j] = dissimilarity
        dm[j, i] = dissimilarity

mean_dissimilarity = np.mean(dm)
std_dissimilarity = np.std(dm)

print("====== dissimilarity matrix ======")
print(dm)

print("Mean Dissimilarity:", mean_dissimilarity)
print("Standard Deviation of Dissimilarity:", std_dissimilarity)

print("creating matrix file...")
np.save("dissimilarity_matrix2.npy", dm)
print ("matrix file created")