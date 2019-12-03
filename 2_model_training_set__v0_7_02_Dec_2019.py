
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

# Logging status to log file
os.chdir(output_path)
prefix_20_hashes = "\n"+"#"*20+"  "
suffix_20_hashes = "  "+"#"*20
file_name = '2_model_random_forest_results_'+datetime.now().strftime("%d_%B_%Y_%H_%M_%S")+'.log'
log_to_file(prefix_20_hashes +"Import libraries completed"+suffix_20_hashes,file_name)  

# Load data
os.chdir(output_path_csvs)
df_ohe = read_file('df.csv')
target = 'Survived'

# Split data into train and test
test_size=0.20
random_state=100
X_train,X_test,y_train,y_test,X,y = split(target,df_ohe,test_size,random_state)

# Train the model
X_train,y_train,feat_importances,important_features_list_new,model = train_model(X_train,X_test,y_train,y_test,X,y,'RF',df_ohe)

# Predict the results
confusion_matrix,classification_report = predict_results(model,X_test,y_test)

# Write results to log file
prefix_20_hashes = "\n"+"#"*20+"  "
suffix_20_hashes = "  "+"#"*20
file_name = '2_model_random_forest_results_'+datetime.now().strftime("%d_%B_%Y_%H_%M_%S")+'.log'
os.chdir(output_path)

log_to_file(prefix_20_hashes +"Feature Name and Importance"+suffix_20_hashes,file_name)  
log_to_file(feat_importances,file_name)

log_to_file(prefix_20_hashes +"Feature Importance order"+suffix_20_hashes,file_name)  
log_to_file(important_features_list_new,file_name)

log_to_file(prefix_20_hashes +"Confusion Matrix"+suffix_20_hashes,file_name)  
log_to_file(confusion_matrix,file_name)

log_to_file(prefix_20_hashes +"Classification Report"+suffix_20_hashes,file_name)
log_to_file(classification_report,file_name)

# decile_analysis(df)

log_to_file("\n ###########  End of Modelling ###################### ",file_name)        
