# %%
import pandas as pd
import numpy as np
from math import log2

# %%
df = pd.DataFrame()
df['chest_pain'] = [1,1,1,0,1,0]
df['male'] = [1,1,0,1,1,1]
df['smokes'] = [0,1,0,0,1,1]
df['exercises'] = [1,0,0,1,1,1]
df['heart_failure'] = [1,1,1,0,1,0]
df

# %%
def information_gain(dataframe, class_label):
    # Calculate entropy of the class label
    class_counts = dataframe[class_label].value_counts()
    class_probabilities = class_counts / len(dataframe)
    class_entropy = -sum(class_probabilities * np.log2(class_probabilities))

    # Calculate entropy of each attribute and information gain
    ig_dict = {}
    for col in dataframe.columns:
        if col == class_label:
            continue
        col_counts = dataframe[col].value_counts()
        col_entropy = 0
        for value in col_counts.index:
            subset = dataframe[dataframe[col] == value]
            value_prob = len(subset) / len(dataframe)
            value_class_counts = subset[class_label].value_counts()
            value_class_probabilities = value_class_counts / len(subset)
            value_class_entropy = -sum(value_class_probabilities * np.log2(value_class_probabilities + (value_class_probabilities==0)))
            col_entropy += value_prob * value_class_entropy
        ig_dict[col] = class_entropy - col_entropy
    return ig_dict

information_gain(df, 'heart_failure')


