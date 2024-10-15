import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import tree
import pickle
from sklearn.metrics import accuracy_score
import seaborn as sns

#Add important column
def Add_column(data, col_index, col_name, col_value) -> pd.DataFrame:
    data.insert(col_index, col_name, col_value)

#Drop needless column
def Drop(data, col_drop) -> pd.DataFrame:
    data = data.drop(columns=col_drop, inplace = True)
#Group culture into larger groups
def Cult_group(data, cult_major, cult_minor) -> pd.DataFrame:
    for i in range(1, 1558):
        Minor = 0
        if not(data.loc[i, 'culture'] in cult_major):
            for clans in cult_minor:
                if data.loc[i, 'culture'] in cult_minor[clans]:
                    data.loc[i, 'culture'] = clans
                    Minor = 1
                #Destroy thr cycle if there was a replacement
                if Minor == 1:
                    break
            #Check Other
            if Minor == 0:
                 data.loc[i, 'culture'] = 'Other'
#Change attributes to 0/1 condition
def Change_attributes(data) -> pd.DataFrame:
    for k in range(1,1558):
      #Make sex attribute
      if (data.loc[k, 'male'] == 0):
          data.loc[k, 'female'] = 1
      
      #Make popularity to 0/1 condition
      if (data.loc[k, 'popularity'] >= 0.5):
          data.loc[k, 'popularity'] = 1
      else:
          data.loc[k, 'popularity'] = 0

      #Make numDeadRelations to 0/1 condition
      if (data.loc[k, 'numDeadRelations'] >= 1):
          data.loc[k, 'numDeadRelations'] = 1
      else:
          data.loc[k, 'numDeadRelations'] = 0

#Rename columns
def Rename(data) -> pd.DataFrame:
    data.rename(columns={'numDeadRelations':'boolDeadRelations'}, inplace = True)
    data.rename(columns={'popularity':'isPopular'}, inplace = True)

#Culture data
def Col_drop():
    col_drop = ['dateOfBirth', 'mother', 'father', 
                'heir', 'spouse', 'isAliveMother', 'isAliveFather', 
                'isAliveHeir', 'isAliveSpouse', 'name', 'age', 'title',
                'house', 'book1', 'book2', 'book3', 'book4', 'book5']
    return col_drop
    
#Culture data
def Cult_major():
    cult_major = ['Northmen', 'Ironborn', 'Free Folk', 'Braavosi', 'Valyrian', 'Dothraki',
                'Ghiscari', 'Dornish', 'Reach', 'Mountain clans', 'Valemen', 'Westerman', 
                'Tyroshi', 'Qartheen']
    return cult_major
    
#Culture data
def Cult_minor():
    cult_minor = {  'Northmen': ['northmen'],
                   'Ironborn':['Ironmen', 'ironborn'],
                   'Free Folk':['Free folk', 'free folk', 'Wildling'],
                   'Braavosi':['Braavos'],
                   'Ghiscari':['Ghiscaricari'],
                   'Dornish':['Dornishmen', 'Dorne'],
                   'Mountain clans':['Vale mountain clans', 'Northern mountain clans'],
                   'Valemen':['Vale'],
                   'Westerman':['Westeros', 'Westermen', 'westermen'],
                   'Qartheen':['Qarth']}
    return cult_minor
    
#Culture data
def Data_feat():
    data_feat = ['culture']
    return data_feat

#Change more columns by OneHotEncoder
def One_hot(data, data_feat) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output = False)
    one_hot_encoded = encoder.fit_transform(data[data_feat])
    one_hot_data = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(data_feat),
                               index = data.index)

#Test_data storage
def Test_data(data_encoded):
    X = data_encoded.drop(columns=['isAlive']).values
    y = data_encoded['isAlive'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return {'X_train' : X_train, 'X_test' : X_test,
            'y_train' : y_train, 'y_test' : y_test}

#Train model and show accuracy
def Model_and_acc(test_data_dict):
    #Data standardization
    scaler = MinMaxScaler()
    scaler.fit(test_data_dict['X_train'])
    X_train_scaled = scaler.transform(test_data_dict['X_train'])
    X_test_scaled = scaler.transform(test_data_dict['X_test'])

    #Train model
    logistic_regression = LogisticRegression(C=1)
    logistic_regression.fit(test_data_dict['X_train'], test_data_dict['y_train'])

    #Final accuracy
    y_pred = logistic_regression.predict(test_data_dict['X_test'])
    accuracy = accuracy_score(test_data_dict['y_test'], y_pred)
    print("Accuracy : %.4f" % accuracy)

    # Download your model to disk
    #filename = 'finalized_model.sav'
    #pickle.dump(clf, open('finalized_model.sav', 'wb'))

    # load the model from disk
    #loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

#Load DataFrames
dftrain = pd.read_csv("https://drive.google.com/uc?id=1XL0VTygpZj-ZAuTNRBgApZTPQyNDnT-v",
                        index_col='S.No')
dftest = pd.read_csv("https://drive.google.com/uc?id=1h99toeF7lZ2I3iJwehgKO-QQmDaOe_O3",
                        index_col = 'S.No')

#Part of calling funtions
Add_column(dftrain, 4, 'female', 0)
Drop(dftrain, Col_drop())
Change_attributes(dftrain)
Cult_group(dftrain, Cult_major(), Cult_minor())
Rename(dftrain)

#Make Encoded DataFrame
encoder = OneHotEncoder(sparse_output = False)
one_hot_encoded = encoder.fit_transform(dftrain[Data_feat()])
one_hot_dftrain = pd.DataFrame(one_hot_encoded,columns = encoder.get_feature_names_out(Data_feat()), 
                                index = dftrain.index)
dftrain_encoded = pd.concat([dftrain, one_hot_dftrain], axis = 1)
dftrain_encoded.drop(columns = Data_feat(), inplace = True)

#Last step
Model_and_acc(Test_data(dftrain_encoded))
