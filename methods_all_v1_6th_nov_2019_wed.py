
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.show()
plt.figure() 
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.externals import joblib

from datetime import datetime

################################################### Writing to Log #################################################################

# Working
# Writing any errors into log file under output
import logging
import logging.handlers
import os

def errors_to_log_file(path):    
    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", path))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)
    
    logging.debug('Stage 1 completed')
    logger.debug('Stage 1 debug completed')
    logger.error('Stage 1 error completed')
    
    try:
        
        print(2/0)
        exit(main())
        
    except Exception:
        logging.exception("Exception in main()")
        exit(1)
        
def read_file(file_name):
    print('file_name:',file_name)
    print('before'+file_name+'.csv'+'after')
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
    print(train_df[[Pclass, Survived]].groupby([Pclass], as_index=False).mean().sort_values(by=Survived, ascending=False))
    
def facegrid(train_df,Survived,Age):
    # Analyze by visualizing data
    g = sns.FacetGrid(train_df, col=Survived)
    print(g.map(plt.hist, Age, bins=20))
    
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
    categoricals = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df[col].fillna(0, inplace=True)
    return df,categoricals        
            
def create_dummy(df,categoricals):
    df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)
    return df_ohe

def train_model(target,algo,df):
    X = df.drop(target,axis = 1)
    y = df[target]
    print("y:")
    print(y.head())
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 100)
    
    if(algo == 'RF'):
        model = RandomForestClassifier(n_estimators = 100)
        
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    return model
    
    def save_model(lr):
#         model_saved = joblib.dump(model, 'model.pkl')
#         print("Model dumped!")
#         return model_saved
    
        # Save your model
        from sklearn.externals import joblib
        joblib.dump(lr, 'model.pkl')
        print("Model dumped!")
        return joblib.dump(lr, 'model.pkl')

    
def write_file(df,path,file_name):
    print('\n current path: \n',os.getcwd())
    df.to_csv(path+file_name+'_'+datetime.now().strftime("%d_%B_%Y_%H_%M_%S"),index = False)
    return df
    
#     def load_model(model = 'model.pkl'):
#         # Load the model that you just saved
#         rf = joblib.load(model)
#         return rf

        
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    lr = joblib.load(model_file_name) # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)    
    
    
    