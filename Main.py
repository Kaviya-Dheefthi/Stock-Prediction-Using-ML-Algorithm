import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import Stock as stck
import Prediction as pred
import PriceForcast as price
import preprocess as prepro

from matplotlib import pyplot as plt



from_date = datetime.datetime.today()
currentDate = time.strftime("%d_%m_%y")
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
fontColor=(255,255,255)
cond=0


window = tk.Tk()
window.title("STOCK PRICE PREDICTION")

 
window.geometry('1280x720')
window.configure(background='#ECECEC')
#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message1 = tk.Label(window, text="STOCK PRICE PREDICTION" ,bg="#3498DB"  ,fg="white"  ,width=50  ,height=3,font=('times', 25, 'bold')) 
message1.place(x=100, y=20)

lbl = tk.Label(window, text="ENTER SYMBOLE",width=20  ,height=2  ,fg="#FDFEFE"  ,bg="#ED5153" ,font=('times', 15, ' bold ') ) 
lbl.place(x=100, y=200)

txt = tk.Entry(window,    bg="#FDFEFE" ,fg="#ED5153",font=('times', 15, ' bold '))
txt.place(x=400, y=200,height=30  ,  width=200)

lbl = tk.Label(window, text="(EX:SBIN)",width=16  ,height=1  ,fg="#FDFEFE"  ,bg="#ED5153" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=250)


lbl4 = tk.Label(window, text="NOTIFICATION : ",width=20  ,fg="#FDFEFE"  ,bg="#ED5153"  ,height=2 ,font=('times', 15, ' bold')) 
lbl4.place(x=100, y=500)

message = tk.Label(window, text="" ,bg="#FDFEFE"  ,fg="#ED5153"  ,width=30  ,height=2, activebackground = "#F4D03F" ,font=('times', 15, ' bold ')) 
message.place(x=400, y=500)


def clear():
	txt.delete(0, 'end')    
	res = ""
	message.configure(text= res)
    
def submit():
	sym=txt.get()
	if sym != "" :
		stck.getPrice(sym)
		print("DataSet Created Successfully")
		res = "DataSet Created Successfully"
		message.configure(text= res)
	else:
		res = "Enter Symble"
		message.configure(text= res)
	print("Submit")
	
def predict():
	print("predict")
	prepro.preprocess(txt.get())
	res = "Prediction Finished"
	message.configure(text= res)
	#pred.Predict()
	
def forecast():
	print("forecast")
	price.Forcast()
	res = "Forcast Finished"
	message.configure(text= res)


  


addst = tk.Button(window, text="SUBMIT", command=submit  ,fg="#FDFEFE"  ,bg="#3498DB"  ,width=15  ,height=2, activebackground = "#ED5153" ,font=('times', 15, ' bold '))
addst.place(x=100, y=600)

trainImg = tk.Button(window, text="PREDICT", command=predict  ,fg="#FDFEFE"  ,bg="#3498DB"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=300, y=600)

detect = tk.Button(window, text="FORECAST", command=forecast  ,fg="#FDFEFE"  ,bg="#3498DB"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
detect.place(x=500, y=600)

quitWindow = tk.Button(window, text="QUIT", command=window.destroy  ,fg="#FDFEFE"  ,bg="#3498DB"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=700, y=600)

clearButton = tk.Button(window, text="CLEAR", command=clear  ,fg="#FDFEFE"  ,bg="#3498DB"   ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=900, y=600)

 
window.mainloop()