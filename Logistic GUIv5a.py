# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9/7/2020

@author: Alun Brain (Dr. Brain Stats)
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import scrolledtext
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import seaborn as sns
import statsmodels.api as sm
#import statsmodels.formula.api as sm

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tkinter import filedialog


import pandas.core.algorithms as algos
from pandas import Series
import re
import traceback

import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
#reduction using RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


Tk().withdraw()

root = tk.Tk()
root.title("Python GUI - Logistic Model")

#900 (width)x 750 (height) is size of box,  +10+10 is position of box onscreen

root.geometry("990x750+30+30")
root.resizable(0, 0)
root.configure(bg='#49A')
#scroll box on top
scr = scrolledtext.ScrolledText(root, width=121, height=20, wrap=tk.WORD)
scr.grid(column=0, columnspan=10)

pd.set_option('display.max_columns', 500)

def getCSV():
    global df1
    global valid
    
    import_file_path = filedialog.askopenfilename()
    df1 = pd.read_csv (import_file_path)

    build,valid=train_test_split(df1,test_size=0.30,random_state=0)    
    df1=build

    combo1['values'] = list(df1)[0:]
    combo1.current(0)

    for i in list(df1)[0:]:
        combo2.insert(tk.END, i)
        
    scr.insert(tk.INSERT, "Data Details : ")
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, df1.dtypes)
    scr.insert(tk.INSERT, "Build contains " + str(df1.shape[0]) + " rows and validation contains " + str(valid.shape[0]) + " rows")
    scr.insert(tk.INSERT, '\n\n')




#def datasplit():
#    global build
#    global valid
#    global df1
#    build,valid=train_test_split(df1,test_size=0.30,random_state=0)      
#    df1=build
#    
#    #br = df1.shape[0] 
#    #vr = valid.shape[0]
#    scr.insert(tk.INSERT, '\n\n')
#    scr.insert(tk.INSERT, "Data has been split, Do not Press again!")
#    scr.insert(tk.INSERT, "Build contains " + str(df1.shape[0]) + " rows and validation contains " + str(valid.shape[0]) + " rows")#
#    scr.insert(tk.INSERT, '\n\n')


def create_window():
    window = tk.Toplevel(root)


def datadets():
    '''
    populate data details
    '''
   
    
    scr.insert(tk.INSERT, "Data Details : ")
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, df1.dtypes)
    scr.insert(tk.INSERT, '\n\n')
    
    
def corrAll():
    global df1
    '''
    Compute Correlation between all variables
    '''
    #remove probbyx if exists
    df1 = df1.drop(['probbyx'], axis=1, errors='ignore')  
    scr.insert(tk.INSERT, "Correlation between All : ")
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, df1.corr())
    scr.insert(tk.INSERT, '\n\n')


def droplotscateg():
    for col in list(df1.select_dtypes(include=['object']).columns):
        if len(df1[col].unique()) >= 10:
            df1.drop(col,inplace=True,axis=1)
            valid.drop(col,inplace=True,axis=1)
    
      # repopulate boxes            
    combo1['values'] = list(df1)[0:]
    combo1.current(0)
    
    combo2.delete(0, tk.END)
    for i in list(df1)[0:]:
        combo2.insert(tk.END, i)
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, 'dropped vategorical variables with 10+ bins')
    scr.insert(tk.INSERT, '\n\n')      



def onehotencoding():
    global df1
    global valid
    df1 = pd.get_dummies(df1, drop_first=True)
    valid = pd.get_dummies(valid, drop_first=True)
     # repopulate boxes            
   
    combo1['values'] = list(df1)[0:]
    combo1.current(0)
    
    combo2.delete(0, tk.END)
    for i in list(df1)[0:]:
        combo2.insert(tk.END, i)   
    
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, 'ONE HOT ENCODING HAS BEEN COMPLETED')
    scr.insert(tk.INSERT, '\n\n')      


def corr3x():
    '''
    remove all correlated variables and repopulate boxes
    '''
    global df1
    global valid
        #remove probbyx if exists
    df1 = df1.drop(['probbyx'], axis=1, errors='ignore')  

    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = df1.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= 0.8)  and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in df1.columns:
                    del df1[colname] # deleting the column from the dataset
                    del valid[colname]
     # repopulate boxes            

    
    combo1['values'] = list(df1)[0:]
    combo1.current(0)
    
    combo2.delete(0, tk.END)
    for i in list(df1)[0:]:
        combo2.insert(tk.END, i)               

    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, 'CORRELATED VARIABLES AT THE 0.8 CUT-OFF HAS BEEN COMPLETED')
    scr.insert(tk.INSERT, '\n\n') 
    
    
    

def corr3():
    '''
    remove all correlated variables and repopulate boxes
    '''
    global df1
        #remove probbyx if exists
    df1 = df1.drop(['probbyx'], axis=1, errors='ignore')  

    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = df1.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= 0.8)  and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in df1.columns:
                    del df1[colname] # deleting the column from the dataset
                    del valid[colname]
     # repopulate boxes            
   
    combo1['values'] = list(df1)[0:]
    combo1.current(0)
    
    combo2.delete(0, tk.END)
    for i in list(df1)[0:]:
        combo2.insert(tk.END, i)        
    
    
    scr.insert(tk.INSERT, '\n\n')
    scr.insert(tk.INSERT, 'CORRELATED VARIABLES AT THE 0.8 CUT-OFF HAS BEEN COMPLETED')
    scr.insert(tk.INSERT, '\n\n') 
    
    
# calclate IV
        
def iv():
    
    def data_vars(df1x, target):
        global iv
       
        def mono_bin(Y, X, n = 5):
            df1x = pd.DataFrame({"X": X, "Y": Y})
            justmiss = df1x[['X','Y']][df1x.X.isnull()]
            notmiss = df1x[['X','Y']][df1x.X.notnull()]
            r = 0
            while np.abs(r) < 1:
                try:
                    d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
                    d2 = d1.groupby('Bucket', as_index=True)
                    r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                    n = n - 1 
                except Exception as e:
                    n = n - 1
        
            if len(d2) == 1:
                n = 2         
                bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
                if len(np.unique(bins)) == 2:
                    bins = np.insert(bins, 0, 1)
                    bins[1] = bins[1]-(bins[1]/2)
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
                d2 = d1.groupby('Bucket', as_index=True)
            
            d3 = pd.DataFrame({},index=[])
            d3["MIN_VALUE"] = d2.min().X
            d3["MAX_VALUE"] = d2.max().X
            d3["COUNT"] = d2.count().Y
            d3["EVENT"] = d2.sum().Y
            d3["NONEVENT"] = d2.count().Y - d2.sum().Y
            d3=d3.reset_index(drop=True)
            
            if len(justmiss.index) > 0:
                d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
                d4["MAX_VALUE"] = np.nan
                d4["COUNT"] = justmiss.count().Y
                d4["EVENT"] = justmiss.sum().Y
                d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                d3 = d3.append(d4,ignore_index=True)
            
            d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
            d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
            d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
            d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
            d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
            d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
            d3["VAR_NAME"] = "VAR"
            d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
            d3 = d3.replace([np.inf, -np.inf], 0)
            d3.IV = d3.IV.sum()
        
            return(d3)
        
        def char_bin(Y, X):
           
            df1x = pd.DataFrame({"X": X, "Y": Y})
            justmiss = df1x[['X','Y']][df1x.X.isnull()]
            notmiss = df1x[['X','Y']][df1x.X.notnull()]    
            df2 = notmiss.groupby('X',as_index=True)
            
            d3 = pd.DataFrame({},index=[])
            d3["COUNT"] = df2.count().Y
            d3["MIN_VALUE"] = df2.sum().Y.index
            d3["MAX_VALUE"] = d3["MIN_VALUE"]
            d3["EVENT"] = df2.sum().Y
            d3["NONEVENT"] = df2.count().Y - df2.sum().Y
            
            if len(justmiss.index) > 0:
                d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
                d4["MAX_VALUE"] = np.nan
                d4["COUNT"] = justmiss.count().Y
                d4["EVENT"] = justmiss.sum().Y
                d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                d3 = d3.append(d4,ignore_index=True)
            
            d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
            d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
            d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
            d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
            d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
            d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
            d3["VAR_NAME"] = "VAR"
            d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
            d3 = d3.replace([np.inf, -np.inf], 0)
            d3.IV = d3.IV.sum()
            d3 = d3.reset_index(drop=True)
        
            return(d3)
        
        
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
        final = (re.findall(r"[\w']+", vars_name))[-1]
        
        x = df1x.dtypes.index
        count = -1
        
        for i in x:
            if i.upper() not in (final.upper()):
                if np.issubdtype(df1x[i], np.number) and len(Series.unique(df1x[i])) > 2:
                    conv = mono_bin(target, df1x[i])
                    conv["VAR_NAME"] = i
                    count = count + 1
                else:
                    conv = char_bin(target, df1x[i])
                    conv["VAR_NAME"] = i            
                    count = count + 1
                    
                if count == 0:
                    iv_df = conv
                else:
                    iv_df = iv_df.append(conv,ignore_index=True)
        
        iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
        iv = iv.reset_index()
      #  return(iv_df,iv)
      #  iv = iv_df.groupby('VAR_NAME').IV.max()
       # print(iv)
    val1 = combo1.get()
    dep_var = val1.split(" ")     
    df1['tagxxc']=df1[dep_var]
    
    if(df1.tagxxc.nunique() ==2):
        
        data_vars(df1, df1['tagxxc'])
        scr.insert(tk.INSERT, '\n\n')
        scr.insert(tk.INSERT, iv)
        scr.insert(tk.INSERT, '\n\n')
    else:
        scr.insert(tk.INSERT, 'Alert!, Check target is binary')
        scr.insert(tk.INSERT, '\n\n')

#while 'predicted' in valy: valy.remove('predicted') 
  
def removeIV():
    global df1 
    global valid 
    global xy
    global values
    '''
    remove all weak variables repopulate boxes
    '''
    val1 = combo1.get()
   # dep_var = val1.split(" ")
    
    keepcol = iv[(iv.IV>=0.1 )]
    keepcol1 = keepcol['VAR_NAME']
    values = keepcol1.values.tolist()
    xy=val1
    
    values.insert(0, xy)
    #only delete if exists
    while 'probbyx' in values: values.remove('probbyx') 
    
    df1 = df1[values]
    valid = valid[values]
     # repopulate boxes            
   
    combo1['values'] = list(df1)[0:]
    combo1.current(0)
    
    combo2.delete(0, tk.END)
    for i in list(df1)[0:]:
        combo2.insert(tk.END, i)              
    
    
def rfe():
    global df1
#reduction using RFE

#You will use RFE with the Logistic Regression classifier to select the top 10 features. 
    df1 = df1.drop(['probbyx'], axis=1, errors='ignore') 
    val1 = combo1.get()
    dep_var = val1.split(" ")
    df1['tagxxc']=df1[dep_var]

    if (df1.tagxxc.nunique() ==2):
        del df1['tagxxc']
        ay = pd.DataFrame(df1[dep_var])
        alun = np.ravel(df1[dep_var])
        ya = list(ay.columns)
        Xa = df1.drop(ya, 1)
       # names = list(Xa.columns)
        scr.insert(tk.INSERT, '\n\n')
        model = LogisticRegression(max_iter=1000)
        rfe = RFE(model, 10)
        fit = rfe.fit(Xa, alun)
        
        ax = pd.DataFrame(fit.ranking_)
        ax.columns = ['rank']
        ax1 = pd.DataFrame(Xa.columns)
        ax1.columns = ['VAR_NAME']
        brain = pd.merge(ax,ax1,left_index=True,right_index=True)
        
        scr.insert(tk.INSERT,  brain)
        # match the rfe ranking with the variable names and get them in order
       
       # scr.insert(tk.INSERT, print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))) )
        scr.insert(tk.INSERT, '\n\n')
    else:
        del df1['tagxxc']
        scr.insert(tk.INSERT, 'Alert!, Select Independent Variable(s) or check target is binary')
        scr.insert(tk.INSERT, '\n\n')
    
def RegAna():
    '''
    Compute Logistic Regression Model
    User select one Depedended Varaible and one or more Independent Variable(s)
    '''
    val1 = combo1.get()
    dep_var = val1.split(" ")
    values = [combo2.get(idx) for idx in combo2.curselection()]
    val2 = ','.join(values)
    ind_var = val2.split(",")
    df1['tagxxc']=df1[dep_var]
    
    if(val2 != "") and (df1.tagxxc.nunique() ==2):
        scr.insert(tk.INSERT, "Logistic Regression between " + val1 + " ~ " + val2 +
                   " : ")
        scr.insert(tk.INSERT, '\n\n')
        x=df1[ind_var]
        x.insert(0, 'Intercept', 1)
        y=df1[dep_var]
        logit_model=sm.Logit(y,x)
        result=logit_model.fit()
        df1['probbyx']=  result.predict(x)
        valx=valid[ind_var]
        valx.insert(0, 'Intercept', 1)
        valid['probbyx']=  result.predict(valx)
        del df1['tagxxc']
        scr.insert(tk.INSERT, result.summary())
        scr.insert(tk.INSERT, '\n\n')
    else:
        del df1['tagxxc']
        scr.insert(tk.INSERT, 'Alert!, Select Independent Variable(s) or check target is binary')
        scr.insert(tk.INSERT, '\n\n')

def GiniPlot():
    '''
    Draw Ginit Plot
    '''
    val1 = combo1.get()
    dep_var = val1.split(" ")
    values = [combo2.get(idx) for idx in combo2.curselection()]
    val2 = ','.join(values)
    ind_var = val2.split(",")


    if(val2 != ""):
        scr.insert(tk.INSERT, '\n\n')
        AUC = roc_auc_score(df1[dep_var], df1['probbyx'])
        Gini = (2 * AUC) -1
        vAUC = roc_auc_score(valid[dep_var], valid['probbyx'])
        vGini = (2 * vAUC) -1
        fpr, tpr, thresholds = roc_curve(df1[dep_var], df1.probbyx)
        vfpr, vtpr, vthresholds = roc_curve(valid[dep_var], valid.probbyx)
        plt.figure()
        plt.plot(fpr, tpr, label='Build Gini = %0.2f' % Gini )
        plt.plot(vfpr, vtpr, label ='Valid Gini = %0.2f' % vGini)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.title("Gini Plot")
        plt.show()
        scr.insert(tk.INSERT, '\n\n')
    else:
        scr.insert(tk.INSERT, 'Alert!, Select Independent Variable(s)')
        scr.insert(tk.INSERT, '\n\n')


def fkitbutton():
    global df1
    global valid
    val1 = combo1.get()
    dep_var = val1.split(" ")
    
    #drop big ctegorical
    for col in list(df1.select_dtypes(include=['object']).columns):
        if len(df1[col].unique()) >= 10:
            df1.drop(col,inplace=True,axis=1)
            valid.drop(col,inplace=True,axis=1)
            
    df1 = df1.drop(['probbyx'], axis=1, errors='ignore')  

    #drop coorelated
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = df1.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= 0.8)  and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in df1.columns:
                    del df1[colname] # deleting the column from the dataset
                    del valid[colname]
                    
    scr.insert(tk.INSERT, "Correlation done ")                
                    
    iv()
    
    scr.insert(tk.INSERT, "IV calcs ")  
    
    #remove iv    
    keepcol = iv[(iv.IV>=0.1 )]
    keepcol1 = keepcol['VAR_NAME']
    values = keepcol1.values.tolist()
    xy=val1
    
    values.insert(0, xy)
    #only delete if exists
    while 'probbyx' in values: values.remove('probbyx') 
    
    df1 = df1[values]
    valid = valid[values]
    scr.insert(tk.INSERT, "removedIV")  

    #on hot encoding
    onehotencoding()
    

    df1 = df1.drop(['probbyx'], axis=1, errors='ignore') 
    df1['tagxxc']=df1[dep_var]

    if (df1.tagxxc.nunique() ==2):
        del df1['tagxxc']
        ay = pd.DataFrame(df1[dep_var])
      #  alun = np.ravel(df1[dep_var])
        ya = list(ay.columns)
        Xa = df1.drop(ya, 1)
        Xa.insert(0, 'Intercept', 1)
       # names = list(Xa.columns)
        scr.insert(tk.INSERT, '\n\n')

       
  #      scr.insert(tk.INSERT, "Logistic Regression between " + val1 + " ~ " + val2 +
   #                " : ")
        scr.insert(tk.INSERT, "Logistic Regression ")
        scr.insert(tk.INSERT, '\n\n')
        logit_model=sm.Logit(ay,Xa)
        result=logit_model.fit()
        df1['probbyx']=  result.predict(Xa)
        valx=valid.drop(ya, 1)
        valx.insert(0, 'Intercept', 1)
        valid['probbyx']=  result.predict(valx)
       
        scr.insert(tk.INSERT, result.summary())
        scr.insert(tk.INSERT, '\n\n')
        scr.insert(tk.INSERT, '\n\n')
        AUC = roc_auc_score(df1[dep_var], df1['probbyx'])
        Gini = (2 * AUC) -1
        vAUC = roc_auc_score(valid[dep_var], valid['probbyx'])
        vGini = (2 * vAUC) -1
        fpr, tpr, thresholds = roc_curve(df1[dep_var], df1.probbyx)
        vfpr, vtpr, vthresholds = roc_curve(valid[dep_var], valid.probbyx)
        plt.figure()
        plt.plot(fpr, tpr, label='Build Gini = %0.2f' % Gini )
        plt.plot(vfpr, vtpr, label ='Valid Gini = %0.2f' % vGini)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.title("Gini Plot")
        plt.show()
        scr.insert(tk.INSERT, '\n\n')

    else:
        del df1['tagxxc']
        scr.insert(tk.INSERT, 'Alert!, Select Independent Variable(s) or check target is binary')
        scr.insert(tk.INSERT, '\n\n')


# import data

browseButton_CSV = ttk.Button(root, text="  -----  Import CSV File  ----    ", command=getCSV)
browseButton_CSV.grid(row=1, column=1,  pady=10)


datadetsx = ttk.Button(root, text="  -Data Details    ", command=datadets)
datadetsx.grid(row=1, column=4,  pady=10)

# split between build and valid

#split = ttk.Button(root, text="70/30 Split ", command=datasplit)
#split.grid(row=1, column=3,  pady=10)



# Combo Box - 1
lbl_sel1 = ttk.Label(root, text="Select Target Variable").grid(row=2, column=0)
ch1 = tk.StringVar()
combo1 = ttk.Combobox(root, width=12, textvariable=ch1)
combo1.grid(row=2, column=1)


# Combo Box - 2

lbl_sel2 = ttk.Label(root, text="Select Independend Variable").grid(row=2, column=2, padx=(10))
ch2 = tk.StringVar()

#frame = Frame(root)

frame = Frame(root)
frame.grid(column=3, row=2)

combo2 = Listbox(frame, width=20, height=6, selectmode=tk.MULTIPLE)
combo2.grid(column=4, row=2, sticky='w')

scrollbar = Scrollbar(frame, orient="vertical")
scrollbar.config(command=combo2.yview)
#ns makes it fill the side of the frame
scrollbar.grid(column=5, row=2, sticky='ns')

combo2.config(yscrollcommand=scrollbar.set)

reg_btn = ttk.Button(root, text="  REGRESSION  ", command=RegAna)
reg_btn.grid(row=2, column=5, pady=10)

corrAll_btn = ttk.Button(root, text="Correlation Between All", command=corrAll)
corrAll_btn.grid(row=7, column=0)

corr3_btn = ttk.Button(root, text="Remove Correlated Vars", command=corr3)
corr3_btn.grid(row=9, column=0, pady=20)

iv_btn = ttk.Button(root, text="IV Stats", command=iv)
iv_btn.grid(row=7, column=1)

remiv_btn = ttk.Button(root, text="Remove the weak", command=removeIV)
remiv_btn.grid(row=9, column=1)

drop_btn = ttk.Button(root, text="Drop it like it's hot", command=droplotscateg)
drop_btn.grid(row=9, column=2, pady=20)


onehot_btn = ttk.Button(root, text="One Hot Encoding", command=onehotencoding)
onehot_btn.grid(row=9, column=3, pady=20)

rfe_btn = ttk.Button(root, text="RFE", command=rfe)
rfe_btn.grid(row=9, column=5, pady=20)



#scatter_btn = ttk.Button(root, text="Scatter Plot (Max. 4 Pairs)",
#                         command=scatterPlot)
#scatter_btn.grid(row=10, column=2, pady=10)

GiniPlot_btn = ttk.Button(root, text="Gini Plots",  command=GiniPlot)
GiniPlot_btn.grid(row=10, column=3, pady=10)

#probPlot_btn = ttk.Button(root, text="Probability Plot", command=probPlot)
#probPlot_btn.grid(row=10, column=5, pady=10)


fkit = ttk.Button(root, text="Fk it",  command=fkitbutton)
fkit.grid(row=10, column=5, pady=10)



def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


exit_btn = ttk.Button(root, text="Exit", command=_quit)
exit_btn.grid(row=12, column=5, padx=10, pady=30)

root.mainloop()

