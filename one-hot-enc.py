import pandas as pd

# Load the data
data = pd.read_csv('train.csv', index_col='Id')


df = pd.get_dummies(data['LotConfig'], dtype=int)
print(df)

columns = ['LotConfig_'+str(i) for i in df.columns]
print(columns)
df.columns = columns
print(df)

data = pd.concat([data, df], axis=1)
print(data.head())
print(data['LotConfig'].head())
data.drop('LotConfig', inplace=True)