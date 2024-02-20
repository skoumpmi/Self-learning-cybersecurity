import pandas as pd

#df = pd.read.csv('datasets/mqtt_dataset.csv')

df = pd.read_csv('datasets/mqtt_dataset.csv', chunksize=10000)
df_new = pd.DataFrame()
for chunk in df:
    print(chunk)
    chunk = chunk.dropna()
    df_new = pd.concat([df_new, chunk]).reset_index(drop = True)
    print(chunk)
    
print(df_new)
