#importing necessary package from python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import csv

from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import Prediction as pred


def preprocess(Stock):
	data = read_csv('Stock.csv', header=0)
	#data = data.dropna()
	print(data)
	data=np.array(data)


	    	
	pred.Predict("Stock.csv",Stock)
