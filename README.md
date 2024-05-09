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

import pandas as pd
data=pd.read_csv('C:\Users\black\Downloads\spam.csv',encoding="Windows-1252")
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35, random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
acc=accuracy_score(y_test, y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
data:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/9ec509f4-82a8-411d-8d46-32320d4f966c)


data.shape:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/62a119d3-fbd5-43bc-afd3-e8e33bbc1674)



x_shape:
![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/310d0493-f627-4e17-82e2-461d0d6dcbf3)



y_shape:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/abb8d48b-1e80-4838-86f2-8e949f29e83a)

x_train:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/db2e00b6-c02c-408f-ae83-e6598538635d)

x_train.shape:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/a7629527-1d1b-409f-ab2c-411ed0c04f6f)

y_pred:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/a8d9f171-a614-44c0-b9d7-d83742981053)

accuracy:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/a7368ce9-eea8-4b5c-8707-c68ec94a6a85)

confusion matrix

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/e1a528ad-b772-44cb-9a82-8190a0907540)

classification report:

![image](https://github.com/Jwalamukhi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145953628/31bde20b-7d41-4eec-9598-49844bd475a7)














## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
