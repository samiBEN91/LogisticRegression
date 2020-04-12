import pandas as pd
import numpy as np
import seaborn as sns
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv("train.csv")
#print (titanic_data.head(10)) #voir les 10 premiers lignes 
print("# Number of passengers in original data:"+str(len(titanic_data.index)))

## Analyzing Data

#sns.countplot(x="Survived",hue="Sex", data=titanic_data)# 0 for not suvived and 1 for survived
sns.countplot(x="Survived",hue="Pclass", data=titanic_data)# nombre des morts suvant les classes (Chepar, lower...we have three class)
plt.show()
titanic_data["Age"].plot.hist()
plt.show()
titanic_data["Fare"].plot.hist()
plt.show()
titanic_data.info()

sns.countplot(x="SibSp", data=titanic_data)
plt.show()

#Data Wrangling

print(titanic_data.isnull())
print(titanic_data.isnull().sum())
sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap="viridis")
plt.show()

sns.boxplot(x="Pclass",y="Age",data=titanic_data)
plt.show()

print (titanic_data.head(5))
titanic_data.drop("Cabin",axis=1,inplace=True)# enlever la colonne Cabin parce qu il contient bcp de Nan
print (titanic_data.head(5))

titanic_data.dropna(inplace=True)#enlever tous les nan
sns.heatmap(titanic_data.isnull(),yticklabels=False, cbar=False)
plt.show()

print(titanic_data.isnull().sum()) #tous sont à zéro
print (titanic_data.head(2))
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
print(sex)
#print(sex.head(5))

embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
print(embark)
#print(embark.head(5))

Pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
print(Pcl)
#print(Pcl.head(5))
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
print (titanic_data)

titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
print(titanic_data)
titanic_data.drop('Pclass',axis=1,inplace=True)
print(titanic_data)


##Train Data
x=titanic_data.drop("Survived",axis=1) #independ variable
y=titanic_data["Survived"]#depend variable
print("Colunm Survived=\n",y)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

print("y_test\n",y_test)
print("taille_y_test",(len(y_test)))

predictions=logmodel.predict(x_test)
print("predictions\n",predictions)
print("taille_prediction",(len(predictions)))
print(predictions[0],predictions[3],predictions[4])

print(classification_report(y_test,predictions))

print (confusion_matrix(y_test,predictions))
print (accuracy_score(y_test,predictions)*100)
#je uis arrete a 35.17
#https://www.youtube.com/watch?v=VCJdg7YBbAQ