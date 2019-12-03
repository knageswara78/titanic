
######################### Import libraries ###################################################
import os
import numpy as np
import pandas as pd
from datetime import datetime

## Uncomment below code to use graphs.
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.show()
# plt.figure() 
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report

import logging
import logging.handlers

from sklearn.base import TransformerMixin

import pickle
from sklearn.externals import joblib

######################### Writing to Log  ###################################################

# Writing any errors into log file under output
def log_to_file(msg,file_name):   
    #logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE",path))
    
    # Removes root:INFO format argument added here.
    logging.basicConfig(format='%(msg)s', filename=file_name,level=logging.DEBUG)    
    logging.info(msg)   
    
# def errors_to_log_file_with_exception(path,msg):
#     handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE",path))
#     formatter = logging.Formatter(logging.BASIC_FORMAT)
#     #handler.setFormatter(formatter)
        
#     #root = logging.getLogger()
#     #root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
#     #root.addHandler(handler)

#     logging.error(msg) # Working

#     try:
#         print(2/4)
#     #exit(main())
#     except Exception:
#         logging.exception("Exception in main()")
#     exit(1)

######################### Missing values treatment  ###################################################   

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

######################### Other Methods  ###################################################  

# Read csv file
def read_file(file_name):
    df = pd.read_csv(file_name)
    return df

# Display all columns
def display_all_columns():
    pd.set_option('display.max_columns',None)

# Replace prefix after reading from db
def replace_prefix(df):
    prefix = df.columns.values[0].split('.')[0]+'.'
    df.columns = df.columns.str.replace(prefix,'')
    return df

# Group and sort.
def groupby_sort(train_df,Pclass,target):
    # Analyze by grouping features
    return(train_df[[Pclass, target]].groupby([Pclass], as_index=False).mean().sort_values(by=target,                 ascending=False))
    
# def facegrid(train_df,target,Age):
#     # Analyze by visualizing data
#     g = sns.FacetGrid(train_df, col=target)
#     #print(g.map(plt.hist, Age, bins=20))
  
# Remove duplicates
def duplicate_remove(df):
    df = df.drop_duplicates(keep = False)
    return df

# Separate numerical and categorical columns
def separate_numeric_categoric(df):    
    cols = df.columns.values    
    num_cols = df._get_numeric_data().columns.tolist()    
    cat_cols = list(set(cols) - set(num_cols)) 
    return cols,cat_cols,num_cols      

# Aggregate data using max or latest. For example if age ,max age will be taken.
def aggregate(df,subset,Max_lifestage):
    subset_Maxcols = pd.DataFrame(subset.groupby('p_icno')[Max_lifestage].agg({max}).reset_index())
    return(subset_Maxcols)

# trend mean standard deviation and Covariance
def trend_mean_std_CoV(df):
    df = df.sort_values(by=['p_inco','file_date'],ascending = True)
    fet_cols = ['trend','mean','std','CoV']
    base_features_ls = pd.DataFrame()
    base_features_ls['p_icno'] = pd.unique(df.p_icno)
    base_features_ls['row_id'] = (df.row_id)
    return(base_features_ls)

def covrtn(x):
    dd = np.asarray(x)
    if len(dd) !=1:
        return dd.std()/dd.mean()
    else:
        return 0

# Continue from here.

# Create dummy variables
def create_dummy(df,categoricals):
    df_ohe = pd.get_dummies(df, columns=categoricals,drop_first=True)  
    return df_ohe

# Split data into train and test data
def split(target,df,test_size,random_state):
    print(target)
    print(df)
    X = df.drop(target,axis = 1)  
    y = df[target]     
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,                                     random_state=random_state)
    return X_train,X_test,y_train,y_test,X,y

# Train the model
def train_model(X_train,X_test,y_train,y_test,X,y,algo,df):    
    if(algo == 'RF'):
        model = RandomForestClassifier(n_estimators = 100)
    model.fit(X_train,y_train)
    
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    
    important_features_dict = {}
    for x,i in enumerate(model.feature_importances_):
        important_features_dict[x]=i
    important_features_list_new = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True)
    print('Most important features: %s' %important_features_list_new)
    
    # New scikit doc code
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")   

    print(" \n Data Set shape is: \n",X.shape[0],X.shape[1])
    for f in range(X.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f],X.columns.values[indices[f]],importances[indices[f]]))
    
    return X_train,y_train,feat_importances,important_features_list_new,model

def predict_results(model,X_test,y_test):
    pred = model.predict(X_test)    
    conf_matrix =confusion_matrix(y_test,pred)
    clas_report = classification_report(y_test,pred)
    return conf_matrix,clas_report

def save_model(model):    
    pickle.dump(model, open('model_rf.pickle','wb'))    
    return pickle.dump(model, open('model_rf.pickle','wb'))
    
def write_file(df,file_name):
    df.to_csv(file_name,index = False)
    return df

############### Only SMOTE  ############################################################
# Over-sampling: SMOTE
def smote(X, y):
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)
    
    plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
    
    return X_sm, y_sm

################  SMOTE and Tomek ##############################################################
# Over-sampling followed by under-sampling
# Now, we will do a combination of over-sampling and under-sampling, using the SMOTE and Tomek links techniques:
def smote_and_tomek(X, y):
    from imblearn.combine import SMOTETomek
    
    smt = SMOTETomek(ratio='auto')
    X_smt, y_smt = smt.fit_sample(X, y)
    
    plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
    
    return X_smt, y_smt

# Split validation into X and y
def split_target(target,df):
    y = df[target]
    X = df.drop(target,axis = 1)        
    return X,y

############### Decile Analysis  ############################################################

# def decile_analysis(df):
# #     df = pd.DataFrame(np.arange(10), columns=['investment'])
#     df.head()
#     df['decile'] = pd.qcut(df['target'], 10, labels=False)
    
#     df.head()
    
#     df['quintile'] = pd.qcut(df['target'], 5, labels=False)
#     df.head()
    
#     # Sorted in increasing order
#     df['quintile'] = pd.qcut(df['target'], 5, labels=np.arange(5, 0, -1))
#     df['decile'] = pd.qcut(df['target'], 10, labels=np.arange(10, 0, -1))
#     df
    
#     return df
# # decile_analysis(df)

# def split_ranges(df):
#     # write code

    
# # Unique sort groupby
# def unique_sort_group(df_max,df_cat):
#     # write code

# #  Returns rows based on p_icno and file_date
# def sort_group1(df,p_icno,file_date):
#      # write code

##################### main  ######################################################################

# if __name__ == '__main__':
#     try:
#         port = int(sys.argv[1]) # This is for a command-line argument
#     except:
#         port = 12345 # If you don't provide any port then the port will be set to 12345
#     lr = joblib.load(model_file_name) # Load "model.pkl"
# #     print ('Model loaded')
#     model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
# #     print ('Model columns loaded')
#     app.run(port=port, debug=True)    
    
    
    