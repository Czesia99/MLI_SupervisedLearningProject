# artificial dataset
# 6 columns / 300 lines

import numpy as np
import pandas as pd

def main():
    print("Creating artificial dataset csv file...")

    datapoints = 300
    cols = 6

    means = np.linspace(1, 10, cols)
    deviations = np.linspace(0.5, 2.5, cols)
    dataset = pd.DataFrame()

    for i in range(cols):
        data = np.random.normal(means[i], deviations[i], datapoints)
        if i == 0:
            data = data.astype(int)
        elif i == 1:
            data = data.astype(float)
        
        dataset[f'Column_{i+1}'] = data
    
    #positively correlated
    dataset['Column_7'] = dataset['Column_1'] + dataset['Column_2']
    #negatively
    dataset['Column_8'] = dataset['Column_3'] - dataset['Column_4']
    #correlation with 0
    dataset['Column_9'] = np.random.normal(0, 1, datapoints)
    dataset.to_csv('artificial_dataset.csv', index=False)

    print("file created !")

if __name__ == "__main__":
    main()