# -*- coding: utf-8 -*-
"""hotel-reservation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19pq5xIUIk7gIg_QPi9mkWIdLh0NvqZPn
"""

import pandas as pd

"""Through this notebook we will be going through how to approach a dataset and decide what model and attribute to chose for working"""

df=pd.read_csv("Hotel Reservations.csv")

df.head()

df.info()

"""the attributes which are objects are boooking_id, booking_status, market_segment_type,room_type_reseved."""

df.isna().sum()

df["booking_status"].unique()

import seaborn as sns

sns.displot(df['arrival_date'])

df['market_segment_type'].unique()

df['room_type_reserved'].unique()

df1=df.drop('Booking_ID',axis=1)

df1.head()

col=['type_of_meal_plan','room_type_reserved','booking_status','market_segment_type']

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in col:
    df1[i]=le.fit_transform(df1[i])

df1.head()

df1.describe()

df1.info()

sns.pairplot(df1)

X=df1.drop('booking_status',axis=1)
Y=df1['booking_status']

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500)
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(X,Y)

rf.fit(xtr,ytr)

ypre=rf.predict(xte)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.base import accuracy_score

print(f1_score(yte,ypre))
print(accuracy_score(yte,ypre))