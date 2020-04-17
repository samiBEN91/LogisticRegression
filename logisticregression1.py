#Predicting if a person would buy life insurnace based on his age 
#using logistic regression

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
df = pd.read_csv("insurance_data.csv")
print(df.head())
print (df.info())
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.9)

print("X_train\n",X_train,"\n", "taille X_train\n",len(X_train) )
print("X_test\n",X_test)
print("y_train\n",y_train)
print("y_test\n",y_test)

model = LogisticRegression()
print("model",model)

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print("y_predicted\n",y_predicted)

classificationreport=classification_report(y_test,y_predicted)
print("classificationreport\n",classificationreport)

confusionmatrix=confusion_matrix(y_test,y_predicted)
print("confusionmatrix\n",confusionmatrix)

accuracyscore=accuracy_score(y_test,y_predicted)*100
print ("accuracyscore\n",accuracyscore)

print("coef du model\n",model.coef_)
print("intercept\n",model.intercept_)
proba_predicted=model.predict_proba(X_test)
print("proba_predicted\n",proba_predicted)


donnee_test_sami = pd.read_csv("donnee_test_sami.csv")
print("donnee_test_sami\n",donnee_test_sami)
prid_donnee_sami=model.predict(donnee_test_sami)
print("prid_donnee_sami\n",prid_donnee_sami)
