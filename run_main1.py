
# python -W ignore foo.py

import warnings
warnings.filterwarnings("ignore")

#export PYTHONWARNINGS="ignore"

## Loading data

import os

current = os.getcwd()
# print('current path: \n', current)

path1 = 'C:/Users/sunitha G/PycharmProjects/Methods_Templates_Modules'
os.chdir(path1)
from methods_all_v1_6th_nov_2019_wed import *
# print('current path: \n', os.getcwd())

os.chdir(current)
print('current path: \n', os.getcwd())

# path2 = 'C:/Users/sunitha G/PycharmProjects/Methods_Templates_Modules/input'
path2 = 'C:/Users/sunitha G/PycharmProjects/Methods_Templates_Modules'
os.chdir(path2)
# print('current path: \n', os.getcwd())


train_df = read_file('train')
test_df = read_file('test')
combine = [train_df, test_df]

# Analyze by describing data
print(train_df.columns.values)

shtid(train_df)

groupby_sort(train_df,'Pclass','Survived')
print('\n')

groupby_sort(train_df,'Sex','Survived')
print('\n')

groupby_sort(train_df,'SibSp','Survived')
print('\n')

groupby_sort(train_df,'Parch','Survived')
print('\n')

facegrid(train_df,'Survived','Age')
print('\n')

facegrid_more(train_df,'Survived','Age','Pclass')
print('\n')

facegrid_pointplot(train_df,'Embarked','Pclass','Survived','Sex')
print('\n')

print('\n before',train_df.shape)
print(train_df.head())

df = duplicate_remove(train_df)

print('\n after',train_df.shape)
print(train_df.head())

print('check1:\n',train_df.columns.values)


train_df =train_df.drop(['SibSp','PassengerId','Pclass','Name','Parch','Ticket','Fare','Cabin','Embarked'],axis =1)
df = train_df

#train_df = train_df.drop('Name',axis =1)

#train_df.drop([['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']],axis =1)


df,categoricals = separate_numeric_categoric(train_df)


print('\n df_ohe \n',categoricals)


df_ohe = create_dummy(df,categoricals)
df = df_ohe
print('\n df_ohe \n',df_ohe)


print('\n final: \n')

print(df.shape)

print('\n',df.head())

model = train_model('Survived','RF',df_ohe)

#save_model(model)

path_output = 'C:\\Users\\sunitha G\\PycharmProjects\\Methods_Templates_Modules\\titanic_all_v1\\output\\'

write_file(df,path_output,'file_name')


path = path_output+"yourapp5.log"

errors_to_log_file(path)








