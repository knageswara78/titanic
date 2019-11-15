# Date and Time  - 13th Nove 2019 Time 10 am.

################################################## # 1. Import libraries ###################################################
import os
import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.show()
# plt.figure() 
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.externals import joblib

from datetime import datetime

import logging
import logging.handlers


import numpy as np
from sklearn.base import TransformerMixin

from sklearn.externals import joblib

################################################### Writing to Log #################################################################

# Working
# Writing any errors into log file under output


def log_to_file(msg,file_name,path):
#     import logging
    #logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE",path))
    logging.basicConfig(format='%(msg)s', filename=file_name,level=logging.DEBUG)  # To remove root:INFO format argument added here.
    
    logging.info(msg)   
    
def errors_to_log_file_with_exception(path,msg):
        handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE",path))
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        #handler.setFormatter(formatter)
        
        #root = logging.getLogger()
        #root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        #root.addHandler(handler)
        
        logging.error(msg) # Working
        
        try:
            print(2/4)
        #exit(main())
        except Exception:
            logging.exception("Exception in main()")
        exit(1)

################################################### Missing values treatment #########################################################

# import pandas as pd
# import numpy as np

# from sklearn.base import TransformerMixin

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

################################################### Other Methods #################################################################

def read_file(file_name):
    print('in read Current 2 :\n',os.getcwd())  
    df = pd.read_csv(file_name+'.csv')
    return df

def shtid(df):
    print(df.shape)
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.describe())
    
def groupby_sort(train_df,Pclass,Survived):
    # Analyze by grouping features
    return(train_df[[Pclass, Survived]].groupby([Pclass], as_index=False).mean().sort_values(by=Survived, ascending=False))
    
def facegrid(train_df,Survived,Age):
    # Analyze by visualizing data
    g = sns.FacetGrid(train_df, col=Survived)
    #print(g.map(plt.hist, Age, bins=20))
    
def facegrid_more(train_df,Survived,Age,Pclass):
    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
    grid = sns.FacetGrid(train_df, col=Survived, row=Pclass, size=2.2, aspect=1.6)
    grid.map(plt.hist, Age, alpha=.5, bins=20)
    grid.add_legend();
    
def facegrid_pointplot(train_df,Embarked,Pclass, Survived,Sex):
    # grid = sns.FacetGrid(train_df, col='Embarked')
    grid = sns.FacetGrid(train_df, row=Embarked, size=2.2, aspect=1.6)
    grid.map(sns.pointplot, Pclass, Survived,Sex, palette='deep')
    grid.add_legend()
    
def duplicate_remove(df):
    df = df.drop_duplicates(keep = False)
    return df

def separate_numeric_categoric(df):
    cols = df.columns.values
    #print(cols)
    num_cols = df._get_numeric_data().columns    
    cat_cols = list(set(cols) - set(num_cols))
    
    #categoricals = []
    #for col, col_type in df.dtypes.iteritems():
        #if col_type == 'O':
            #categoricals.append(col)
        #else:
            #df[col].fillna(0, inplace=True)
    return cat_cols,num_cols      
            
def create_dummy(df,categoricals):
    df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)
    df_ohe =df_ohe.drop(['Sex_nan','Sex_male','Embarked_S','Embarked_nan'],axis =1)
    return df_ohe

def split(target,df):
    X = df.drop(target,axis = 1)
    y = df[target]
#     print("y:")
#     print(y.head)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 100)
    return X_train,X_test,y_train,y_test,X,y
    
def train_model(X_train,X_test,y_train,y_test,X,y,algo,df):
#     X = df.drop(target,axis = 1)
#     y = df[target]
# #     print("y:")
# #     print(y.head)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 100)
    
    if(algo == 'RF'):
        model = RandomForestClassifier(n_estimators = 100)
        
    model.fit(X_train,y_train)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print(feat_importances)
    
    important_features_dict = {}
    for x,i in enumerate(model.feature_importances_):
        important_features_dict[x]=i
    important_features_list_new = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True)
    print('Most important features: %s' %important_features_list_new)


#     print(feat_importances.nlargest(4).plot(kind='barh'))

    #print("\n New important variabels: \n',model.feature_importances_)
    pred = model.predict(X_test)
    rf = model
    
#     feature_importances = pd.DataFrame(rf.feature_importances_,
#                                    index = df.columns,
#                                    columns=['Survived']).sort_values('Survived',ascending=False)
#     print(feature_importances)
#     #print(confusion_matrix(y_test,pred))
    #print(classification_report(y_test,pred)) # ,confusion_matrix(y_test,pred),classification_report(y_test,pred)
    conf_matrix =confusion_matrix(y_test,pred)
    clas_report = classification_report(y_test,pred)
    
    return X_train,y_train,feat_importances,important_features_list_new,model,conf_matrix,clas_report

#     return X_train,y_train,model,confusion_matrix(y_test,pred),classification_report(y_test,pred),feature_importances
     
    def save_model(lr):
#         model_saved = joblib.dump(model, 'model.pkl')
#         print("Model dumped!")
#         return model_saved
    
        # Save your model
#         from sklearn.externals import joblib
        joblib.dump(lr, 'model.pkl')
#         print("Model dumped!")
        return joblib.dump(lr, 'model.pkl')

    
def write_file(df,path,file_name):
#     print('\n current path: \n',os.getcwd())
    df.to_csv(path+file_name+'_'+datetime.now().strftime("%d_%B_%Y_%H_%M_%S"),index = False)
    return df
    
#     def load_model(model = 'model.pkl'):
#         # Load the model that you just saved
#         rf = joblib.load(model)
#         return rf

#########################################   Only SMOTE     #######################################################################
# Over-sampling: SMOTE
# SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.


# We'll use ratio='minority' to resample the minority class.

def smote(X, y):
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)
    
    plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
    
    return X_sm, y_sm

#########################################   SMOTE and Tomek #######################################################################

# Over-sampling followed by under-sampling
# Now, we will do a combination of over-sampling and under-sampling, using the SMOTE and Tomek links techniques:

def smote_and_tomek(X, y):
    from imblearn.combine import SMOTETomek
    
    smt = SMOTETomek(ratio='auto')
    X_smt, y_smt = smt.fit_sample(X, y)
    
    plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
    
    return X_smt, y_smt
    
################################### Decile Analysis  #################################################################################

def decile_analysis(df):
#     df = pd.DataFrame(np.arange(10), columns=['investment'])
    df.head()
    df['decile'] = pd.qcut(df['investment'], 10, labels=False)
    
    df.head()
    
    df['quintile'] = pd.qcut(df['investment'], 5, labels=False)
    df.head()
    
    # Sorted in increasing order
    df['quintile'] = pd.qcut(df['investment'], 5, labels=np.arange(5, 0, -1))
    df['decile'] = pd.qcut(df['investment'], 10, labels=np.arange(10, 0, -1))
    df
    
    return df
# decile_analysis(df)

########################################### main   ###################################################################################






if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    lr = joblib.load(model_file_name) # Load "model.pkl"
#     print ('Model loaded')
    model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
#     print ('Model columns loaded')
    app.run(port=port, debug=True)    
    
    
    