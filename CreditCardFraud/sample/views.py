from django.shortcuts import render
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Create your views here.
def cdf(request):
	if request.method == "POST":
		t = float(request.POST['time'])
		a = float(request.POST['amount'])
		dataset = pd.read_csv('sample/creditcard.csv')
		y= dataset['Class']
		x= dataset[['Time','Amount']]
		sc = StandardScaler()
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)
		x_train = sc.fit_transform(x_train)
		x_test = sc.transform(x_test)
		classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
		classifier.fit(x_train, y_train)
	
		#regressor.fit(x,y)
		
		y_pred=classifier.predict([[t,a]])
		
		#b=b.split('.')
		if y_pred[0] == 0:
			v="Not Fraud"
		else:
			v="Fraud"
		#print(v,y_pred1[0])
		#print("Accuracy = %f",a)
		#d ={'f':t,'s':a}
		#y_pred = regressor.predict(x_test)
		#accuracy=accuracy_score(y,y_pred,normalize=True)
		# print(accuracy)


		return render(request,'scy/cform.html',{'f':v})
	return render(request,'scy/cform.html')

def description(request):
	return render(request,'scy/description.html') 

def decision():
	if request.method == "POST":
		dataset = pd.read_csv('sample/creditcard.csv')
		y= dataset['Class']
		x= dataset.drop(columns=['Class'],axis=1)
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)
		regressor = DecisionTreeRegressor(random_state = 0)
		regressor.fit(x_train,y_train)
		y_pred = regressor.predict(x_test)
		a=accuracy_score(y_test,y_pred)
		print(a)
	return a 
	# yamuna 1 min downloadd chaystha git bash


def random(request):
	if request.method == "POST":
		dataset = pd.read_csv('sample/creditcard.csv')
		y= dataset['Class']
		x= dataset.drop(columns=['Class'],axis=1)
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)
		sc = StandardScaler()
		x_train = sc.fit_transform(x_train)
		x_test = sc.transform(x_test)
		classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		b=accuracy_score(y_test,y_pred)
		print(b)
		return b

def logistic(request):
	if request.method == "POST":
		dataset = pd.read_csv('sample/creditcard.csv')
		y= dataset['Class']
		x= dataset.drop(columns=['Class'],axis=1)
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)
		classifier = LogisticRegression()
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		c=accuracy_score(y_test,y_pred)
		print(c)
		return c
def Result(request):
	t = request.POST['time']
	a =  request.POST['amount']
	r=decision(t,a)
	# s=random(request)
	# t=logistic(request)
	return render(request,'scy/cform.html',{'v':r})









