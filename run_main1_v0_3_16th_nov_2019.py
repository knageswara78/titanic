
################################################## # 1. Import libraries ###################################################
from datetime import datetime
import os

import pandas as pd



################################################### 1. Declare global variables ############################################

pd.set_option('display.max_columns', None)


# path_user = 'C:\\Users\\sunitha G'
path_user = ''
date_stamp = datetime.now().strftime("%d_%B_%Y_%H_%M_%S")
path_current = '\\dm_offshore\\Modules\\projects\\sample'

path_absolute = path_user+path_current
print('\n absolute path: \n',path_absolute)

path_absolute_date_stamp = path_user+path_current+'\\output\\logs\\'+date_stamp+'.log'
print('\n absolute date stamp path: \n',path_absolute_date_stamp)

path = path_current

print('Current 1 :\n',os.getcwd())
################################################### 1. Import custom methods ###################################################

# os.chdir(r"C:\Users\sunitha G\dm_offshore\methods_framework\methods")
# print('Current 3 :\n',os.getcwd())
print('Current 11 :\n',os.getcwd())     
from methods_all_v0_3_16th_Nov_2019 import *


# os.chdir(r".\output\results")
# os.chdir(path_absolute+r"\output\results")
msg = "\n ######################################## Import libraries completed ########################################\n"
file_name = 'myapp.log'
path = path_absolute+'/output/results/'
log_to_file(msg,file_name,path)


################################################### Supress warnings ###################################################

# python -W ignore foo.py

#export PYTHONWARNINGS="ignore"

# import warnings
# warnings.filterwarnings("ignore")

################################################### 1. Loading data ###################################################
# os.chdir(path_absolute+r"\input")
# os.chdir(r"C:\Users\sunitha G\dm_offshore\Modules\projects\sample\input")
train_df = read_file('train')
test_df = read_file('test')
combine = [train_df, test_df]



################################################### Begin Missing values treatment #########################################################

# print(train_df)

train_df = DataFrameImputer().fit_transform(train_df)

# print(train_df)

# ################################################### End Missing values treatment #########################################################

msg = "\n ########################################  Loading data completed ########################################\n"
# log_to_file(msg,file_name,path)

#log_to_file(train_df,file_name,path)

# # ################################################### 1. Visualize data ###################################################

# # # Analyze by describing data
# log_to_file('\n ########################################  Columns of data set : ########################################\n',file_name,path)
# log_to_file(train_df.columns.values,file_name,path)            
# # print(train_df.columns.values)

# shtid(train_df)

# ################################################### Aggregate data ###################################################

Pclass = groupby_sort(train_df,'Pclass','Survived')

Sex = groupby_sort(train_df,'Sex','Survived')


SibSp = groupby_sort(train_df,'SibSp','Survived')

Parch = groupby_sort(train_df,'Parch','Survived')


msg = "\n ########################################  Aggregate data completed ########################################\n"
# log_to_file(msg,file_name,path)


# # facegrid(train_df,'Survived','Age')
# # print('\n')

# # facegrid_more(train_df,'Survived','Age','Pclass')
# # print('\n')

# # facegrid_pointplot(train_df,'Embarked','Pclass','Survived','Sex')
# # print('\n')


df = duplicate_remove(train_df)

# print('\n after',train_df.shape)
# print(train_df.head())

# print('check1:\n',train_df.columns.values)


train_df =train_df.drop(['PassengerId','Ticket','Name','Cabin'],axis =1)
df = train_df

#train_df = train_df.drop('Name',axis =1)

#train_df.drop([['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']],axis =1)


cols,cat_cols,num_cols = separate_numeric_categoric(train_df)

# path = 'C:\\Users\\sunitha G\\dm_offshore\\Modules\\projects\\sample\\output\\results\\'+datetime.now().strftime("%d_%B_%Y_%H_%M_%S")+'.log'

# os.chdir(r"C:\Users\sunitha G\dm_offshore\Modules\projects\sample\output\results")
print('categ Current 3 :\n',os.getcwd())

log_to_file("\n ########################################  All columns #######################################\n",'myapp.log',path)
log_to_file(cols,'myapp.log',path)

log_to_file("\n ########################################  Categorical columns #######################################\n",'myapp.log',path)
log_to_file(cat_cols,'myapp.log',path)

log_to_file("\n ########################################  Numerical columns #######################################\n",'myapp.log',path)
log_to_file(num_cols,'myapp.log',path)

# print('\n df_ohe \n',categoricals)

print('\n i before cat_cols : \n',cat_cols) # ['Embarked', 'Sex']

df_ohe = create_dummy(df,cat_cols)

log_to_file('\n ############################# Data after dummy ##################################### \n','myapp.log',path)
log_to_file(df_ohe.head(),'myapp.log',path)


X_train,X_test,y_train,y_test,X,y = split('Survived',df_ohe)

X_train,y_train,feat_importances,important_features_list_new,model,confusion_matrix1,classification_report1 = train_model(X_train,X_test,y_train,y_test,X,y,'RF',df_ohe)

   
msg = confusion_matrix1
file_name = 'myapp.log'

log_to_file("\n ######################################## Feature Importance : #######################################\n",'myapp.log',path)
log_to_file(feat_importances,'myapp.log',path)

log_to_file("\n ######################################## Feature Importance order : #######################################\n",'myapp.log',path)
log_to_file(important_features_list_new,'myapp.log',path)

 
log_to_file("\n ########################################  Confusion Matrix : #######################################\n",'myapp.log',path)
log_to_file(msg,'myapp.log',path)

msg = classification_report1
file_name = 'myapp.log'
 
log_to_file("\n ########################################  Classification Report ######################################\n",'myapp.log',path)
log_to_file(msg,'myapp.log',path)
           

# # print('\n confusion_matrix1 \n',confusion_matrix1)
# # print('\n classification_report1 \n',classification_report1)

# # #save_model(model)

# # path_output = 'C:\\Users\\sunitha G\\PycharmProjects\\Methods_Templates_Modules\\titanic_all_v1\\output\\'

# # write_file(df,path_output,'file_name')

log_to_file("\n ####################  End of program Date time stamp: ######################################\n",'myapp.log',path)
log_to_file(datetime.now().strftime("%d_%B_%Y_%H_%M_%S"),'myapp.log',path)
        
# decile_analysis(df)
            
