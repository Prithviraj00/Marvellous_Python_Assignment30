import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def BankCaseStudy(datapath):
    df = pd.read_csv(datapath)
    
    print("Data Loaded Successfully :")
    print(df.head())
    print(df.shape)
    
    print("Null Values in Data:\n", df.isnull().sum())
    
    df.replace('unknown', np.nan, inplace=True)
    print("Values after replacing 'unknown':")
    print(df.isnull().sum())
    print(df.head())
    
    for col in df.select_dtypes(object):
        df[col].fillna(df[col].mode()[0],inplace= True)
        
    print(df.head())
    print("Statistical Summary:")
    print(df.describe())
    
    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])
        
    print(df.head())
    
    x = df.drop(columns=['y'])
    
    y = df['y']
    
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    
    logisticRegression(x_scale,y)
    Randomforest(x_scale,y)
    
    
def logisticRegression(x_scale,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = LogisticRegression()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Logistic Regression Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
  
def Randomforest(x_scale,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = RandomForestClassifier(n_estimators=100,max_depth=7, random_state=42)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Random Forest Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)  
    
def main():
    BankCaseStudy("bank-full.csv")

if __name__ == "__main__":
    main()