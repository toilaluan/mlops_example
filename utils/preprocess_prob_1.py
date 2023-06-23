import pandas as pd

def preprocess(data: pd.DataFrame):
    data = data.drop('feature1', axis=1)
    data = pd.get_dummies(data, columns=['feature2'])
    return data