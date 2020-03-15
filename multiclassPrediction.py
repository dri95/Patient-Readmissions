import pandas as pd
import numpy as np

readmission = pd.read_csv("C2T1_Train.csv")
dstest = pd.read_csv("C2T1_Test.csv")
dstest = dstest.rename(columns={'encounter_id':'encounter_id2','patient_nbr':'patient_nbr2'})

#############preprocessing##########################################################################################
def preprocess(dataset):

   dataset = dataset.replace('?',np.nan)
   
   def miss_val(data):
       if dataset.isnull().values.any():
          miss = data.isnull().sum().sort_values(ascending=False)
          return miss
       else: print('There are no missing values in the Dataset')
       
    
   miss_val(dataset)
   
   dataset = dataset.drop(['weight','payer_code','medical_specialty'], axis = 1)
    
   dataset.dropna(subset=['race'], inplace=True)
   
   def unique(data):
       for c in list(data.columns):
           n = data[c].unique()
           if len(n)<=11:
              print(c)
              print(n)
           else:
              print(c + ': ' +str(len(n)) + ' unique values')
   
   unique(dataset)
   
   dataset.groupby('gender').size()
   dataset = dataset[dataset.gender != 'Unknown/Invalid']
           
   dataset = dataset.drop_duplicates(subset = ['patient_nbr2'], keep = 'first')
   
   dataset = dataset.drop(['diag_2','diag_3'], axis = 1)
   
   dataset.dropna(subset=['diag_1'],inplace=True)
   
   miss_val(dataset)
   unique(dataset)
   
   dataset = dataset.drop(['examide','citoglipton','glimepiride-pioglitazone' ,'metformin-rosiglitazone'], axis = 1)
   dataset.dtypes
   
   
   dataset['change'] = dataset['change'].replace('Ch', 1)
   dataset['change'] = dataset['change'].replace('No', 0)
   
   dataset['diabetesMed'] = dataset['diabetesMed'].replace('Yes', 1)
   dataset['diabetesMed'] = dataset['diabetesMed'].replace('No', 0)
   
   
   drugs = ['metformin','repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
          'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide','pioglitazone',
          'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone','tolazamide', 
          'insulin', 'glyburide-metformin', 'glipizide-metformin',
          'metformin-pioglitazone']
   
   for col in drugs:
       dataset[col] = dataset[col].apply(lambda x: 0 if (x == 'No') else 1)
   
   unique(dataset)
   dataset.dtypes
   
   
   def ICD9 (i):
       i = str(i)
       if (i[0].isnumeric() == True):
           i = pd.to_numeric(i) 
           if (i >= 390 and i < 460) or (np.floor(i) == 785):
              return "Circulatory"
           elif (i >= 460 and i < 520) or (np.floor(i) == 786):
              return "Respiratory"
           elif (i >= 520 and i < 580) or (np.floor(i) == 787):
              return "Digestive"
           elif (np.floor(i) == 250):
              return "Diabetes"
           elif (i >= 800 and i < 1000):
              return "Injury	"
           elif (i >= 710 and i < 740):
              return "Musculoskeletal"
           elif (i >= 580 and i < 630) or (np.floor(i) == 788):
              return "Genitourinary"
           elif (i >= 140 and i < 240):
              return "Neoplasms"
           else:
              return "Other"
       else: # if the code does not begin with a number
           return "Other"
   
   dataset["diag_1"] = dataset["diag_1"].apply(ICD9)
   
   dataset.groupby('diag_1').size()
   
   dataset.dtypes
   
   
   dataset['A1Cresult'] = dataset['A1Cresult'].replace('>7', 'high')
   dataset['A1Cresult'] = dataset['A1Cresult'].replace('>8', 'high')
   dataset['A1Cresult'] = dataset['A1Cresult'].replace('Norm', 'Normal')
   dataset['A1Cresult'] = dataset['A1Cresult'].replace('None', 'Not Performed') 
   
   
   dataset['max_glu_serum'] = dataset['max_glu_serum'].replace('>200', 'high')
   dataset['max_glu_serum'] = dataset['max_glu_serum'].replace('>300', 'high')
   dataset['max_glu_serum'] = dataset['max_glu_serum'].replace('Norm', 'Normal')
   dataset['max_glu_serum'] = dataset['max_glu_serum'].replace('None', 'Not Performed')
   
   
   dataset = dataset.loc[~dataset.discharge_disposition_id.isin([11,13,14,19,20,21])]
   
   
   dataset['admission_type_id'] = dataset['admission_type_id'].apply(lambda x: 'Emergency' if (x == 1 or x == 2 or x == 7) 
                                                                            else( 'Unavalible' if (x == 6 or x == 5 or x == 8) else 'Elective'))   

   
   dataset['admission_source_id'] = dataset['admission_source_id'].apply(lambda x: 'Emergency_room' if (x == 7) 
                                                                            else( 'Ph_Cl Referral' if (x == 1 or x == 2) else 'Other'))
                                                                            
   dataset['discharge_disposition_id'] = dataset['discharge_disposition_id'].apply(lambda x: 'Discharge to home' if (x == 1) else 'Other')
   
   
   dataset.groupby('discharge_disposition_id').size()
   
   unique(dataset)
   dataset.dtypes
   
   ########################################
   
   dataset=dataset.astype('object')
   
   num=['number_emergency','num_procedures','number_diagnoses','num_lab_procedures',
        'number_outpatient', 'number_inpatient', 'num_medications','time_in_hospital']
   
   dataset[num] = dataset[num].astype('int64')
   numeric = list(set(list(dataset._get_numeric_data().columns)))
   
   
   
   hot_col = ['race','age','gender','discharge_disposition_id','admission_source_id', 
              'admission_type_id','diag_1','max_glu_serum','A1Cresult']
   
   onehot = dataset[hot_col]
   onehotdum = pd.get_dummies(onehot)
   #ll = list(onehotdum.columns)
   
   dataset = pd.concat([onehotdum, dataset],axis=1)
   dataset.drop(hot_col,axis=1, inplace=True)
   dataset.info()
   
   
   dataset[num].skew()
   names=['number_emergency', 'number_outpatient', 'number_inpatient']
   
   for i in names:
      dataset[i] = np.log1p(dataset[i])
   
   
   
   from sklearn.preprocessing import StandardScaler
   dataset[numeric] = StandardScaler().fit_transform(dataset[numeric])
   import scipy as sp
   dataset = dataset[(np.abs(sp.stats.zscore(dataset[numeric])) < 3).all(axis=1)]
   
   dataset['readmitted'] = dataset['readmitted'].replace('>30', 2)
   dataset['readmitted'] = dataset['readmitted'].replace('<30', 1) 
   dataset['readmitted'] = dataset['readmitted'].replace('NO', 0)
   
   #dataset['readmitted'] = dataset['readmitted'].apply(lambda x: 0 if x == 2 else x)
   #dataset.groupby('readmitted').size()
   return dataset

trainrem = preprocess(readmission) 
trainrem = trainrem.drop(['encounter_id2','patient_nbr2'], axis = 1)
 
testremep = preprocess(dstest)
testrem = testremep.drop(['encounter_id2','patient_nbr2'], axis = 1)

##################################################################################################################################
Y = trainrem['readmitted']
X = trainrem.loc[:, trainrem.columns != "readmitted"]
test_input = testrem.loc[:, testrem.columns != "readmitted"]


######################################LR######################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X_train, x_val, Y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=0)

################################################

from imblearn.over_sampling import SMOTE

from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
smo = SMOTE(random_state=20)
X_new, Y_new = smo.fit_sample(X, Y)
print('New dataset shape {}'.format(Counter(Y_new)))

X_new = pd.DataFrame(X_new, columns = list(X.columns))

X_train, x_val, Y_train, y_val = train_test_split(X_new, Y_new, test_size=0.20, random_state=21)
from sklearn.ensemble import RandomForestClassifier
forrest = RandomForestClassifier(n_estimators = 10, max_depth=25, criterion = "gini", min_samples_split=10)
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(forrest, X_train, Y_train, cv=10))))
forrest.fit(X_train, Y_train)

pred= forrest.predict(x_val)

from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_auc_score
print("Accuracy is {:f}".format(accuracy_score(y_val, pred)))
print("Precision is {:f}".format(precision_score(y_val, pred,average='micro')))
print("Recall is {:f}".format(recall_score(y_val, pred,average='micro')))
#print("AUC is {:f}".format(roc_auc_score(y_val, pred,average='micro')))


from sklearn.metrics import confusion_matrix
labels = ['<30','No','>30']
cm = confusion_matrix(y_val, pred )
cm = pd.DataFrame(cm, index=labels, columns=labels)

import seaborn as sns
import matplotlib.pyplot as plt
cm_plot = sns.heatmap(cm, annot=True, cmap='summer')
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

predtest= forrest.predict(test_input)

pp = pd.DataFrame(predtest)

ep=['encounter_id2','patient_nbr2','readmitted']
testremep['readmitted'] = pp
predictest = testremep[ep]
predictest['readmitted'] = predictest['readmitted'].replace(2, '>30')
predictest['readmitted'] = predictest['readmitted'].replace(1,'<30') 
predictest['readmitted'] = predictest['readmitted'].replace(0,'NO')

predictest = predictest.rename(columns={'encounter_id2':'encounter_id','patient_nbr2':'patient_nbr'})

predictest.to_csv('multi.csv', header=True, index=True)

























