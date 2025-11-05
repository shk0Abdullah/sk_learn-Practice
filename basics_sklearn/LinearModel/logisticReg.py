import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression


df = sns.load_dataset("tips")
print(df.head())

model = LogisticRegression()
X = df[["total_bill", "size", "tip"]]
Y = df["smoker"]
model.fit(X,Y)
print(model.predict(pd.DataFrame([[35,2,0]],columns=["total_bill", "size", "tip"])))