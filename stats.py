import pandas as pd


df = pd.read_csv("runs/out_comp.csv")

average_per_id = df.groupby('Name').mean()
average_per_id.to_csv("out_out_out.csv", sep=',', index=True, encoding='utf-8')
