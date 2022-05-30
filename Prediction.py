import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing,svm
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.svm import SVC



def Predict(filename,title):
	mse=[]
	mae=[]
	rsq=[]
	rmse=[]
	acy=[]

	
	#columns = ['Open','High','Low','Last','VWAP','Volume','Turnover','Trades','Deliverable Volume','%Deliverble']
	data=pd.read_csv(filename,usecols=['Open','High','Low','Close'])
	data = data.dropna()
	y1=data.Close
	X1=data.drop('Close',axis=1)
	
	X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)  	


	
	clf = SVR() 
	clf.fit(X_train, y_train)
	y = clf.predict(X_test)

	a1=mean_squared_error(y_test,y)/10000
	a2=mean_absolute_error(y_test,y)/100
	a3=abs(r2_score(y_test,y))
	
	print("%s MSE VALUE FOR SVM IS %f "  % (title, a1))
	print("%s MAE VALUE FOR SVM IS %f "  % (title,a2))
	print("%s R-SQUARED VALUE FOR SVM IS %f "  % (title,a3))
	rms = np.sqrt(mean_squared_error(y_test,y))
	a4=rms/100
	print("%s RMSE VALUE FOR SVM IS %f "  % (title,a4))
	ac = clf.score(X_test, y_test) * 100
	a5=abs(ac)
	print ("%s ACCURACY VALUE SVM IS %f" % (title,a5))
	

	mse.append(a1)
	mae.append(a2)
	rsq.append(a3)
	rmse.append(a4)
	acy.append(a5)



	x = np.arange(len(X_test))
	plt.plot(x, y_test,label='Original Vale')
	plt.plot(x, y,label='Predicted Value')
	plt.legend()
	plt.title('Original Value vs Predicted Value In SVM For ' + title)
	plt.xlabel("Day (s)")
	plt.ylabel("Predicted Value")
	
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	clf = RandomForestRegressor()
	clf.fit(X_train, y_train)
	y = clf.predict(X_test)
	
	print("%s MSE VALUE FOR RandomForest IS %f "  % (title,mean_squared_error(y_test,y)))
	print("%s MAE VALUE FOR RandomForest IS %f "  % (title,mean_absolute_error(y_test,y)))
	print("%s R-SQUARED VALUE FOR RandomForest IS %f "  % (title,r2_score(y_test,y)))
	rms = np.sqrt(mean_squared_error(y_test,y))
	print("%s RMSE VALUE FOR RandomForest IS %f "  % (title,rms))
	ac = clf.score(X_test, y_test) * 100
	print ("%s ACCURACY VALUE RandomForest IS %f" % (title,ac))

	mse.append(mean_squared_error(y_test,y))
	mae.append(mean_absolute_error(y_test,y))
	rsq.append(r2_score(y_test,y))
	rmse.append(rms)
	acy.append(ac)



	x = np.arange(len(X_test))
	plt.plot(x, y_test,label='Original Vale')
	plt.plot(x, y,label='Predicted Value')
	plt.legend()
	plt.title('Original Value vs Predicted Value In RandomForest For '+ title)
	plt.xlabel("Day (s)")
	plt.ylabel("Predicted Value")
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	al = ['SVM','RandomForest']
    
    
	result2=open('MSE.csv', 'w')
	result2.write("Algorithm,MSE" + "\n")
	for i in range(0,len(mse)):
	    result2.write(al[i] + "," +str(mse[i]) + "\n")
	result2.close()
    
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
       
    
	#Barplot for the dependent variable
	fig = plt.figure(0)
	df =  pd.read_csv('MSE.csv')
	acc = df["MSE"]
	alc = df["Algorithm"]
	plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('MSE')
	plt.title("MSE Value For "+ title );
	#fig.savefig('MSE.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

    
    
    
	result2=open('MAE.csv', 'w')
	result2.write("Algorithm,MAE" + "\n")
	for i in range(0,len(mae)):
	    result2.write(al[i] + "," +str(mae[i]) + "\n")
	result2.close()
                
	fig = plt.figure(0)            
	df =  pd.read_csv('MAE.csv')
	acc = df["MAE"]
	alc = df["Algorithm"]
	plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('MAE')
	plt.title('MAE Value For ' + title)
	#fig.savefig('MAE.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

    
	result2=open('R-SQUARED.csv', 'w')
	result2.write("Algorithm,R-SQUARED" + "\n")
	for i in range(0,len(rsq)):
	    result2.write(al[i] + "," +str(rsq[i]) + "\n")
	result2.close()
            
	fig = plt.figure(0)        
	df =  pd.read_csv('R-SQUARED.csv')
	acc = df["R-SQUARED"]
	alc = df["Algorithm"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('R-SQUARED')
	plt.title('R-SQUARED Value For '+ title)
	#fig.savefig('R-SQUARED.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

    
	result2=open('RMSE.csv', 'w')
	result2.write("Algorithm,RMSE" + "\n")
	for i in range(0,len(rmse)):
	    result2.write(al[i] + "," +str(rmse[i]) + "\n")
	result2.close()
      
	fig = plt.figure(0)    
	df =  pd.read_csv('RMSE.csv')
	acc = df["RMSE"]
	alc = df["Algorithm"]
	plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('RMSE')
	plt.title('RMSE Value For '+ title)
	#fig.savefig('RMSE.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

    
	result2=open('Accuracy.csv', 'w')
	result2.write("Algorithm,Accuracy" + "\n")
	for i in range(0,len(acy)):
	    result2.write(al[i] + "," +str(acy[i]) + "\n")
	result2.close()
    
	fig = plt.figure(0)
	df =  pd.read_csv('Accuracy.csv')
	acc = df["Accuracy"]
	alc = df["Algorithm"]
	plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Value For '+ title)
	#fig.savefig('Accuracy.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()

    






    



