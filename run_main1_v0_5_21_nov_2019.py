
################################################## # 1. Import libraries ###################################################
from datetime import datetime
import os

import pandas as pd

################################################### 1. Declare global variables ############################################

target = 'Survived'
columns_to_drop = ['PassengerId','Ticket','Name','Cabin']

methods_file_name = 'methods_all_v0_5_21_Nov_2019'
is_inside_citrix = 'No'  #'Yes'

if(is_inside_citrix == 'Yes'):
    methods_path = r"/data/r_home/dm_offshore/methods_framework/methods"
    input_path = r"/data/r_home/dm_offshore/methods_framework/projects/sample/input"
    output_path = r"/data/r_home/dm_offshore/methods_framework/projects/sample/output/results"
else:
    methods_path =  r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\code"
    input_path =  r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\input"
    output_path = r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\output\results" 
     
    


################################################### 1. Declare global variables ############################################

pd.set_option('display.max_columns', None)

################################################### 1. Import custom methods ###################################################
os.chdir(methods_path) 

print('Importing methods from below path :\n',os.getcwd())     
from methods_all_v0_5_21_Nov_2019 import *

msg = "\n ######################################## Import libraries completed ########################################\n"
file_name = datetime.now().strftime("%d_%B_%Y_%H_%M_%S")+'.log'

os.chdir(output_path)

log_to_file(msg,file_name)


################################################### Supress warnings ###################################################

# python -W ignore foo.py

#export PYTHONWARNINGS="ignore"

# import warnings
# warnings.filterwarnings("ignore")

################################################### 1. Loading data ###################################################
os.chdir(input_path)
print('Reading files from below path:\n',os.getcwd())  
train_df = read_file('train')
print(train_df.shape)
train_df.head()
test_df = read_file('test')
combine = [train_df, test_df]



################################################### Begin Missing values treatment #########################################################

# print(train_df)

train_df = DataFrameImputer().fit_transform(train_df)

# print(train_df)

# ################################################### End Missing values treatment #########################################################

msg = "\n ########################################  Loading data completed ########################################\n"
# log_to_file(msg,file_name)

#log_to_file(train_df,file_name)

# # ################################################### 1. Visualize data ###################################################

# # # Analyze by describing data
# log_to_file('\n ########################################  Columns of data set : ########################################\n',file_nameh)
# log_to_file(train_df.columns.values,file_name)            
# # print(train_df.columns.values)

# shtid(train_df)

# ################################################### Aggregate data ###################################################
train_df.head()
Pclass = groupby_sort(train_df,'Pclass',target) 

Sex = groupby_sort(train_df,'Sex',target)


SibSp = groupby_sort(train_df,'SibSp',target)

Parch = groupby_sort(train_df,'Parch',target)


msg = "\n ########################################  Aggregate data completed ########################################\n"
# log_to_file(msg,file_name)


# # facegrid(train_df,target,'Age')
# # print('\n')

# # facegrid_more(train_df,target,'Age','Pclass')
# # print('\n')

# # facegrid_pointplot(train_df,'Embarked','Pclass',target,'Sex')
# # print('\n')


df = duplicate_remove(train_df)

# print('\n after',train_df.shape)
# print(train_df.head())

# print('check1:\n',train_df.columns.values)

train_df =train_df.drop(columns_to_drop,axis =1)
df = train_df

#train_df = train_df.drop('Name',axis =1)

#train_df.drop([['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']],axis =1)


cols,cat_cols,num_cols = separate_numeric_categoric(train_df)

os.chdir(output_path)


log_to_file("\n ########################################  All columns #######################################\n",file_name)
log_to_file(cols,file_name) 

log_to_file("\n ########################################  Categorical columns #######################################\n",file_name)
log_to_file(cat_cols,file_name)

log_to_file("\n ########################################  Numerical columns #######################################\n",file_name)
log_to_file(num_cols,file_name)

# print('\n df_ohe \n',categoricals)

df_ohe = create_dummy(df,cat_cols)

log_to_file('\n ############################# Data after dummy ##################################### \n',file_name)
log_to_file(df_ohe.head(),file_name)


X_train,X_test,y_train,y_test,X,y = split(target,df_ohe)

X_train,y_train,feat_importances,important_features_list_new,model,confusion_matrix1,classification_report1 = train_model(X_train,X_test,y_train,y_test,X,y,'RF',df_ohe)

   
msg = confusion_matrix1
file_name = 'myapp.log'

log_to_file("\n ######################################## Feature Importance : #######################################\n",file_name)
log_to_file(feat_importances,file_name)

log_to_file("\n ######################################## Feature Importance order : #######################################\n",file_name)
log_to_file(important_features_list_new,file_name)

 
log_to_file("\n ########################################  Confusion Matrix : #######################################\n",file_name)
log_to_file(msg,file_name)

msg = classification_report1
 
log_to_file("\n ########################################  Classification Report ######################################\n",file_name)
log_to_file(msg,file_name)
           

# # print('\n confusion_matrix1 \n',confusion_matrix1)
# # print('\n classification_report1 \n',classification_report1)

# # #save_model(model)

# # write_file(df,path_output,'file_name')

log_to_file("\n ###########  End of program Date time stamp: "+datetime.now().strftime("%d_%B_%Y_%H_%M_%S")+" ###################### ",file_name)

        
# decile_analysis(df)
            
