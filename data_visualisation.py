import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("~/titanic/train.csv")


import seaborn

seaborn.set()
survived_class=dataset[dataset['Survived']==1]['Pclass'].value_counts()
dead_class=dataset[dataset['Survived']==0]['Pclass'].value_counts()
df_class = pd.DataFrame([survived_class,dead_class])
df_class.index = ['Survived','Died']
df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Class")
plt.show(10)
Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100
Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100
Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100
print("Percentage of Class 1 that survived:" ,round(Class1_survived),"%")
print("Percentage of Class 2 that survived:" ,round(Class2_survived), "%")
print("Percentage of Class 3 that survived:" ,round(Class3_survived), "%")

# display table
from IPython.display import display
display(df_class)

# -------------------Survived/Died by SEX------------------------------------

Survived = dataset[dataset.Survived == 1]['Sex'].value_counts()
Died = dataset[dataset.Survived == 0]['Sex'].value_counts()
df_sex = pd.DataFrame([Survived, Died])
df_sex.index = ['Survived', 'Died']
df_sex.plot(kind='bar', stacked=True, figsize=(5, 3), title="Survived/Died by Sex")
plt.show(1)
female_survived = df_sex.female[0] / df_sex.female.sum() * 100
male_survived = df_sex.male[0] / df_sex.male.sum() * 100
print("Percentage of female that survived:", round(female_survived), "%")
print("Percentage of male that survived:", round(male_survived), "%")

# display table
from IPython.display import display

display(df_sex)
