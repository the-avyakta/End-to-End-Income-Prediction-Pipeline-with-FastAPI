import pandas as pd


def load_data(path):
    # path = '../../Data/adult-census.csv'
    data= pd.read_csv(path)
    df = pd.DataFrame(data)

    X = df.drop(['income','fnlwgt'], axis=1)
    y = df['income'].map({
        '<=50K': 0,
        '>50K': 1
    })
    

    return X,y