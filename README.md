# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1 . Import pandas module and import the required data set.

2 . Find the null values and count them.

3 . Count number of left values.

4 . From sklearn import LabelEncoder to convert string values to numerical values.

5 . From sklearn.model_selection import train_test_split.

6 . Assign the train dataset and test dataset.

7 . From sklearn.tree import DecisionTreeClassifier.

8 . Use criteria as entropy.

9 . From sklearn import metrics.

10 . Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HEMANTH KUMAR R
RegisterNumber:  212223040065
*/
```

```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![370544375-17b12b41-7b04-4aa5-a76a-807cb1bc7a93](https://github.com/user-attachments/assets/3a9db33c-88be-4534-9485-88f8b48b5dd5)

![370544330-735253e2-ed0d-4f7b-aaf2-684c5f705fe7](https://github.com/user-attachments/assets/9121eb4a-5516-41bc-89b7-f33ff7cfc8d4)

![370544341-09355f87-8e36-4476-94fa-0ee3e1dd39ac](https://github.com/user-attachments/assets/0ab2ff15-4e47-4dcb-a70e-52869eb70783)

![370544348-21d36d20-df70-4d58-8c2c-daadb21674b5](https://github.com/user-attachments/assets/426765e0-58bf-46dd-a096-b0dcdd8ad11c)

![370544353-70e37a99-ca09-4770-88ab-836452100cc7](https://github.com/user-attachments/assets/c3240937-504e-4292-9224-4b853b2fedfc)

![370544359-4fa523ff-c8ee-428c-9beb-cf5b5b1c3bd3](https://github.com/user-attachments/assets/4383a426-f977-429a-8789-9b27d3b9a405)

![370544365-25c08290-fd4a-4e86-bacc-bbf8e5356749](https://github.com/user-attachments/assets/cf9c337a-cc13-4bb0-a4ae-ca803c6f3d7e)

![370544370-7503edd4-6e1c-47ec-b13e-847a3168867d](https://github.com/user-attachments/assets/b7d2183f-6fc2-47c2-9b8a-ed54f767760c)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
