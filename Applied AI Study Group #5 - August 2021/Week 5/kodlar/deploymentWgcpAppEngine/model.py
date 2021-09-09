import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# DATA FOR PRED
df=pd.read_csv("data/data.csv")
print(df.head())

logreg=LogisticRegression(solver='liblinear')


X=df.iloc[:,:8]


y=df[["Outcome"]]

X=np.array(X)
y=np.array(y)

logreg.fit(X,y.reshape(-1,)) #reshape 1-d array
joblib.dump(logreg,"model")