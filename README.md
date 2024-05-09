# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load and handle the dataset, separating features and target variables.
2.Data Splitting: Split the dataset into training and testing sets.
3.Feature Engineering: Transform text data into numerical feature vectors.
4.Model Building and Evaluation: Initialize SVM classifier, train the model, and predict target labels for evaluation.



## Program:

### Program to implement the SVM For Spam Mail Detection..
### Developed by:Jwalamukhi S 
### RegisterNumber:212223040079
```
import chardet
file='C:\Users\black\Downloads\spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('C:\Users\black\Downloads\spam.csv',encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred
```

## Output:
head():
![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/b5caaaa7-4e20-4a6b-8f0c-88b82c5997dd)

info():
![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/19ac04b1-bca9-4845-a136-7dafbfdebe11)

isnull().sum():
![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/108b112b-ca5c-473e-b679-1ca6dfd6cbb7)

predicted values:
![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/c678117f-72e4-406a-8426-d913885b1fc0)







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
