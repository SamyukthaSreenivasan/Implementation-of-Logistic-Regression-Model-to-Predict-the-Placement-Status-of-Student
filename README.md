# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 

## Program:
```
Developed by: Samyuktha S
RegisterNumber: 212222240089
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
# Placement Data
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/3c0cd4cc-3f4f-44ea-b777-60cb31b490a4)

# Salary Data
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/e930e9b4-b5e2-4e57-a38b-05107d92cd88)

# Checking the null() function
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/a0955fef-d69a-4926-98d0-e6c9a47c2fbb)

# Data Duplicate 
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/7bf51b24-0782-4c98-afba-5cf91227a732)

# Print Data
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/fd8c8015-b304-443c-b045-f81817407bc5)

# Data-Status
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/013624fd-a58d-4235-9f98-289788fa9173)

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/a9b4c894-4feb-4423-aa83-dba95b47d44d)

# Y_prediction Array
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/450eefbe-457a-4843-9511-f3bafe38946c)

# Accuracy value
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/1f3b35cc-505f-4865-b9bc-cbdb6280e0be)

# Confusion Array
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/6e1cdaf7-5851-4c6a-8328-127f460698dc)

# Classification Report
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/1d5a71ab-05bc-42b4-a1f9-b82f01be910f)

# Prediction of LR
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119475703/500a9e75-e1a5-4d97-9c92-5e8f4dbbb1d4)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
