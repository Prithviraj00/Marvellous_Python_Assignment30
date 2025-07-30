import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def BankCaseStudy(datapath):
    df = pd.read_csv(datapath)
    
    print("Data Loaded Successfully :")
    #print(df.head())
    print(df.shape)
    
    print("Null Values in Data:\n", df.isnull().sum())
    
    df.replace('unknown', np.nan, inplace=True)
    print("Values after replacing 'unknown':")
    print(df.isnull().sum())
    #print(df.head())
    
    for col in df.select_dtypes(object).columns:
        df[col].fillna(df[col].mode()[0],inplace= True)
        
    #print(df.head())
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
    KNN(x_scale,y)
    Randomforest(x_scale,y)
    
      
def logisticRegression(x_scale,y):
    
    print("\nLogistic Regression Result :\n")
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = LogisticRegression()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    y_proba = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Logistic Regression Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    roc_auc = roc_auc_score(y_test,y_proba)
    print ("ROC-AUC:", roc_auc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Logistic Regression Confusion Matrix ")
    plt.show()  
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def KNN(x_scale,y):
    print("\n KNN Classifier Result :\n")
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    y_proba = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("KNN Classifie Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    roc_auc = roc_auc_score(y_test,y_proba)
    print ("ROC-AUC:", roc_auc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("KNN Classifier Confusion Matrix ")
    plt.show() 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - KNN Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()
    
  
def Randomforest(x_scale,y):
    
    print("\n Random Forest Classifier  Result :\n")
    
    x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2,random_state=42)
    
    model = RandomForestClassifier(n_estimators=100,max_depth=7, random_state=42)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    y_proba = model.predict_proba(x_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Random Forest Classifier Accuracy is :",accuracy_score(y_test,y_pred)*100)
    
    cm = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix :",cm)
    
    roc_auc = roc_auc_score(y_test,y_proba)
    print ("ROC-AUC:", roc_auc)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Classifier Confusion Matrix ")
    plt.show() 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest Classifier')
    plt.legend()
    plt.grid(True)
    plt.show() 
    
def main():
    BankCaseStudy("bank-full.csv")

if __name__ == "__main__":
    main()
