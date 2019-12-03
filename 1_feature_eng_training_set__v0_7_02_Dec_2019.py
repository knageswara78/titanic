
# Import libraries
from datetime import datetime
import os
import sys
import pandas as pd

# Define paths
# methods_path = r"/data/r_home/dm_offshore/methods_framework/methods"
# input_path = r"/data/r_home/dm_offshore/methods_framework/projects/sample/input"
# output_path_csvs = r"/data/r_home/dm_offshore/methods_framework/projects/sample/output/results/csvs"
# output_path = r"/data/r_home/dm_offshore/methods_framework/projects/sample/output/results"

methods_path =  r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\code"
input_path =  r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\input"
output_path_csvs = r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\output\csvs"  
output_path = r"C:\Users\sunitha G\dm_offshore\methods_framework\projects\sample\output\results"   

# Import custom methods  
sys.path.append(methods_path)
from methods_all__v0_7_02_Dec_2019 import *

pd.set_option('display.max_columns', None)

# Write status to log file
os.chdir(output_path)
prefix_20_hashes = "\n"+"#"*20+"  "
suffix_20_hashes = "  "+"#"*20
file_name = '1_data_process_feature_engineering_'+datetime.now().strftime("%d_%B_%Y_%H_%M_%S")+'.log'
log_to_file(prefix_20_hashes +"Import libraries completed"+suffix_20_hashes,file_name)  

# Load data
# df = 'train.csv'
# target = 'Survived'
# columns_to_drop = ['PassengerId','Ticket','Name','Cabin']

df = 'train.csv'
target = 'Survived'
columns_to_drop = ['PassengerId','Ticket','Name','Cabin']
 
os.chdir(output_path)

#Supress warnings
# python -W ignore foo.py
#export PYTHONWARNINGS="ignore"
# import warnings
# warnings.filterwarnings("ignore")

#Loading data
os.chdir(input_path)
df = read_file(df)

# Create train and validation sets.
train_set = df.iloc[:600,:]
validation_set = df.iloc[601:890,:]

df = train_set

test_df = read_file('test.csv')
combine = [df, test_df]
log_to_file(prefix_20_hashes +"Loading data completed"+suffix_20_hashes,file_name)  

# Missing values treatment
df = DataFrameImputer().fit_transform(df)

# #Analyze by describing data   
log_to_file(prefix_20_hashes +"Shape of Data"+suffix_20_hashes,file_name)  
log_to_file(df.shape,file_name)

log_to_file(prefix_20_hashes +"Head of Data"+suffix_20_hashes,file_name)  
log_to_file(df.head(),file_name)

log_to_file(prefix_20_hashes +"Tail of Data"+suffix_20_hashes,file_name)  
log_to_file(df.tail(),file_name)

log_to_file(prefix_20_hashes +"Info of Data"+suffix_20_hashes,file_name)  
log_to_file(df.info(),file_name)

log_to_file(prefix_20_hashes +"Describe Data"+suffix_20_hashes,file_name)  
log_to_file(df.describe(),file_name)

# facegrid(df,target,'Age')
# print('\n')

# Duplicate removal
df = duplicate_remove(df)

df =df.drop(columns_to_drop,axis =1)
X_df = df.drop(target,axis =1)

# Separate numeric and categorical columns
all_cols,cat_cols,num_cols = separate_numeric_categoric(X_df)

# Aggregate data
for i in all_cols:
    result = groupby_sort(df,i,target)    
    log_to_file(prefix_20_hashes +"Group by"+suffix_20_hashes,file_name)  
    log_to_file(result,file_name) 

os.chdir(output_path)
log_to_file(prefix_20_hashes +"All columns"+suffix_20_hashes,file_name)  
log_to_file(all_cols,file_name) 

log_to_file(prefix_20_hashes +"Categorical columns"+suffix_20_hashes,file_name)  
log_to_file(cat_cols,file_name)

log_to_file(prefix_20_hashes +"Numerical columns"+suffix_20_hashes,file_name)  
log_to_file(num_cols,file_name)

# Create Dummy varibles for categorical varaibles.
df_ohe = create_dummy(df,cat_cols)

log_to_file(prefix_20_hashes +"All columns after dummy variable creation"+suffix_20_hashes,file_name)  
log_to_file(df_ohe.head(),file_name)

# Write final file
os.chdir(output_path_csvs)
write_file(df_ohe,file_name = 'df.csv')

log_to_file(prefix_20_hashes +"End of Data Processing"+suffix_20_hashes,file_name)  

print("End of Data Processing")  
