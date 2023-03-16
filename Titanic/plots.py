'''
                                            T.C. Sakarya Üniversitesi Bilgisayar ve Bilişim Sistemleri Fakültesi
                                                            Bilgisayar Mühendsiliği Bölümü

                                                            Büyük Veriye Giriş Dersi Proje Ödevi
                                                    Ödev Konusu: Titanic - Machine Learning from Disaster

                                                            Deniz Berfin Taştan / B181210010 / 1-B
                                                        Mustafa Melih Tüfekcioğlu / B191210004 / 1-A

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train['Survived'].value_counts().plot(kind='bar').set(xlabel="Survived", ylabel="Number of passengers")
plt.show()

train['Pclass'].value_counts().plot(kind='bar').set(xlabel="Pclass", ylabel="Number of passengers")
plt.show()

train['SibSp'].value_counts().plot(kind='bar').set(xlabel="Number of siblings / spouses aboard the Titanic", ylabel="Number of passengers")
plt.show()

train['Parch'].value_counts().plot(kind='bar').set(xlabel="Number of parents / children aboard the Titanic", ylabel="Number of passengers")
plt.show()

train['Sex'].value_counts().plot(kind='bar').set(xlabel='sex', ylabel='Number of passengers')
plt.show()

plt.hist(train['Age'])
plt.xlabel('Age')
plt.show()

train['Embarked'].value_counts().plot(kind='bar').set(xlabel="Pclass", ylabel="Number of passengers")
plt.show()

train.corr()
sns.heatmap(train.corr(),cmap="YlGnBu", annot=True)

sns.countplot(x=train['Pclass'],hue=train['Survived'],data=train)
plt.show()

sns.countplot(x=train['Sex'],hue=train['Survived'],data=train)
plt.show()

sns.barplot(x=train['Survived'],y=train['Fare'])
plt.show()

plt.figure(figsize=(12,6))
sns.kdeplot(x="Age",data=train[train['Survived']==0].dropna(),fill=True,alpha=1,color="red",label="Not Survived")
sns.kdeplot(x="Age",data=train[train['Survived']==1].dropna(),fill=True,alpha=0.5,color="blue",label="Survived")
plt.legend()
plt.show()

sns.countplot(x=train['Embarked'],hue=train['Survived'],data=train)
plt.show()