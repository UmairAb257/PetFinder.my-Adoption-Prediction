# # Data Fields
# # # * PetID - Unique hash ID of pet profile
# # # * AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
# # # * Type - Type of animal (1 = Dog, 2 = Cat)
# # # * Name - Name of pet (Empty if not named)
# # # * Age - Age of pet when listed, in months
# # # * Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
# # # * Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
# # # * Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
# # # * Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
# # # * Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
# # # * Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
# # # * MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
# # # * FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
# # # * Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# # # * Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# # # * Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# # # * Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
# # # * Quantity - Number of pets represented in profile
# # # * Fee - Adoption fee (0 = Free)
# # # * State - State location in Malaysia (Refer to StateLabels dictionary)
# # # * RescuerID - Unique hash ID of rescuer
# # # * VideoAmt - Total uploaded videos for this pet
# # # * PhotoAmt - Total uploaded photos for this pet
# # # * Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.
# # # ## AdoptionSpeed
# # # * Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted. The values are determined in the following way: 
# # # * 0 - Pet was adopted on the same day as it was listed. 
# # # * 1 - Pet was adopted between 1 and 7 days (1st week) after being listed. 
# # # * 2 - Pet was adopted between 8 and 30 days (1st month) after being listed. 
# # # * 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed. 
# # # * 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).
# # # 
# # # # Images
# # # For pets that have photos, they will be named in the format of PetID-ImageNumber.jpg. Image 1 is the profile (default) photo set for the pet. For privacy purposes, faces, phone numbers and emails have been masked.
# # # 
# # # ## Image Metadata
# # # We have run the images through Google's Vision API, providing analysis on Face Annotation, Label Annotation, Text Annotation and Image Properties. You may optionally utilize this supplementary information for your image analysis.
# # # 
# # # File name format is PetID-ImageNumber.json.
# # # 
# # # Some properties will not exist in JSON file if not present, i.e. Face Annotation. Text Annotation has been simplified to just 1 entry of the entire text description (instead of the detailed JSON result broken down by individual characters and words). Phone numbers and emails are already anonymized in Text Annotation.
# # # 
# # # Google Vision API reference: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate
# # # 
# # # ## Sentiment Data
# # # We have run each pet profile's description through Google's Natural Language API, providing analysis on sentiment and key entities. You may optionally utilize this supplementary information for your pet description analysis. There are some descriptions that the API could not analyze. As such, there are fewer sentiment files than there are rows in the dataset.
# # # 
# # # File name format is PetID.json.
# # # 
# # # Google Natural Language API reference: https://cloud.google.com/natural-language/docs/basics
# # # 
# # # What will change in the 2nd stage of the competition?
# # # In the second stage of the competition, we will re-run your selected Kernels. The following files will be swapped with new data:
# # # 
# # # test.zip including test.csv and sample_submission.csv
# # # test_images.zip
# # # test_metadata.zip
# # # test_sentiment.zip
# # # In stage 2, all data will be replaced with approximately the same amount of different data. The stage 1 test data will not be available when kernels are rerun in stage 2.


import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

from pandas.io.json import json_normalize

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#os.listdir("C:/Users/Umair/Desktop/ml/petfinder")
#imtest_path= os.listdir('C:/Users/Umair/Desktop/ml/petfinder/test_metadata')
#imtest_names=os.listdir('C:/Users/Umair/Desktop/ml/petfinder/test_metadata')

test = pd.read_csv("../input/test/test.csv")
train = pd.read_csv("../input/train/train.csv")
breed = pd.read_csv("../input/breed_labels.csv")
color = pd.read_csv("../input/color_labels.csv")
state = pd.read_csv("../input/state_labels.csv")
sent_path= '../input/train_sentiment/'

#print("Dimensions of train: {}".format(train.shape))
#print("Dimensions of test: {}".format(test.shape))

#print("Dimensions of breed: ", breed.shape)
#print("Dimensions of color: ", color.shape)
#print("Dimensions of state: ", state.shape)

#train = train[['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'PureBreed', 'Gender', 'Color1', 'Color2', 'Color3', 'ColorCount', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'VideoAmt', 'Description', 'DescriptionCount', 'PetID', 'PhotoAmt', 'AdoptionSpeed', 'NamesVal']]
#train.head(3)


sent_name=os.listdir('../input/train_sentiment')[3]
sent_file = sent_path + sent_name
# sent_file
sent_json=pd.read_json(sent_file, orient='index', typ='series')
# sent_json
json_normalize(sent_json.sentences)
json_normalize(sent_json.entities)
sent_ds=json_normalize(sent_json.documentSentiment)
sent_ds.magnitude[0]
txt_df = pd.DataFrame(columns=['PetID','magnitude','score','language'])
# txt_df
# sent_name.split('.')[0]
# train[train['PetID']=='d81afce4f']

sent_row=pd.DataFrame(data=[sent_name.split('.')[0],sent_json.language,sent_ds.magnitude[0],sent_ds.score[0]], index=['PetID','magnitude','score','language']).transpose()
# sent_row
txt_df.append(sent_row)
sent_json.language
#train['PetID']



def fill_txt_df(sent_path) :
    sent_names=os.listdir(sent_path)
    txt_df = pd.DataFrame(columns=['PetID','magnitude','score','language'])
    for sent_name in sent_names :
        sent_file = sent_path + sent_name
        # print(curr_file)
        sent_json=pd.read_json(sent_file, orient='index', typ='series')
        sent_ds=json_normalize(sent_json.documentSentiment)
        sent_row=pd.DataFrame(data=[sent_name.split('.')[0],sent_ds.magnitude[0],sent_ds.score[0],sent_json.language],
                 index=['PetID','magnitude','score','language']).transpose()
        txt_df=txt_df.append(sent_row)
    return txt_df


txt_df_train = fill_txt_df('../input/train_sentiment/')
# txt_df_train.describe()

txt_df_test = fill_txt_df('../input/test_sentiment/')
# txt_df_test.describe()

#traintxt_data=txt_df_train.append(txt_df_test)
# txt_df_train.describe()
# txt_df_train.dtypes
# txt_df_train.isna().sum()
#print(txt_df_train.info())
#print(txt_df_test.info())


def cols_to_numeric(df, col_names):
    for k in range(0,len(col_names)) :
        df[col_names[k]]=pd.to_numeric(df[col_names[k]],errors='coerce')
    return df
##############
col_names=['magnitude','score']
txt_train = cols_to_numeric(txt_df_train, col_names)
txt_test = cols_to_numeric(txt_df_test, col_names)


print(txt_train)
print("**************************************************************************************************")
print(txt_test)


imtrain_path= '../input/train_metadata/'
imtrain_names=os.listdir('../input/train_metadata')
len(imtrain_names)


imtrain_name=os.listdir('../input/train_metadata')[432]
#imtrain_name
imtrain_file = imtrain_path + imtrain_name
imtrain_file

imtrain_json=pd.read_json(imtrain_file, orient='index', typ='series')
imtrain_json

jla=json_normalize(imtrain_json.labelAnnotations)
jla

jla['description'].str.cat(sep=', ')

jla[jla['score']==jla['score'].max()]

jpa=json_normalize(imtrain_json.imagePropertiesAnnotation)
jpa.iloc[0]

jpa.iloc[0][0]

len(jpa.iloc[0][0])

k=2

jpa.iloc[0][0][k]['color']

jpa.iloc[0][0][k]['score']

jpa.iloc[0][0][k]['pixelFraction']

df_jpa=pd.DataFrame(columns=['color','score','pixelFraction'])
df_jpa

data=[str(jpa.iloc[0][0][k]['color']),jpa.iloc[0][0][k]['score'],jpa.iloc[0][0][k]['pixelFraction']]
pd.DataFrame(data=data,index=['color','score','pixelFraction']).transpose()

df_jpa=pd.DataFrame(columns=['color','score','pixelFraction'])
for k in range(0,len(jpa.iloc[0][0])) :
    curr_data=[str(jpa.iloc[0][0][k]['color']),jpa.iloc[0][0][k]['score'],jpa.iloc[0][0][k]['pixelFraction']]
    curr_row=pd.DataFrame(data=curr_data,index=['color','score','pixelFraction']).transpose()
    df_jpa=df_jpa.append(curr_row)
df_jpa['prod']=df_jpa['score']*df_jpa['pixelFraction']
df_jpa

len(df_jpa[df_jpa['prod']==df_jpa['prod'].max()])

jcha=json_normalize(imtrain_json.cropHintsAnnotation)
jcha.iloc[0]

jcha.iloc[0][0]

jcha.iloc[0][0][0]['boundingPoly']

jcha.iloc[0][0][0]['confidence']

jcha.iloc[0][0][0]['importanceFraction']


# Forked: (Pets Adoption Simple (pandas + random forest))
# #  Thought: Is it possible to summarise all this data? 
# #  The idea is to create a dataframe to join later with df_all. As there are many images per pet, when we will join all the data we will have to choose one of the image rows, using a function like max() or sum().
# #  First, some useful secondary functions:
# #


###*****************************************************************###

def not_empty(s) :
    if len(s)>=1 : 
        return s[0]
    else :
        return ''
    
###*****************************************************************###

def extr_jla_info(imtrain_json) :
    dfr=pd.DataFrame(columns=['score','description'])
    try :
        jlas=json_normalize(imtrain_json.labelAnnotations)
        imtrain_data=[jlas[jlas['score']==jlas['score'].max()]['score'][0],
                   jlas['description'].str.cat(sep=',')]
        imtrain_row=pd.DataFrame(data=imtrain_data,index=['score','description']).transpose()
        dfr=dfr.append(imtrain_row)
    except:
        print('Line Skipped in JLA')
        dfr=pd.DataFrame(data=[0,0], index=['score','description']).transpose()
    return dfr
    
###*****************************************************************###

def extr_jpa_info(imtrain_json) :
    dfr=pd.DataFrame(columns=['color','score','pixelFraction','prod'])
    try :
        jpas=json_normalize(imtrain_json.imagePropertiesAnnotation)
        for k in range(0,len(jpas.iloc[0][0])) :
            imtrain_data=[str(jpas.iloc[0][0][k]['color']),
                   jpas.iloc[0][0][k]['score'],
                   jpas.iloc[0][0][k]['pixelFraction'],
                   jpas.iloc[0][0][k]['score']*jpas.iloc[0][0][k]['pixelFraction']]
            imtrain_row=pd.DataFrame(data=imtrain_data,index=['color','score','pixelFraction','prod']).transpose()
            dfr=dfr.append(imtrain_row)
        dfr=dfr[dfr['prod']==dfr['prod'].max()]    
    except :
        print('Line Skipped in JPA')
        dfr=pd.DataFrame(data=['',0,0,0], index=['color','score','pixelFraction','prod']).transpose()
    return dfr
    
###*****************************************************************###

def extr_jcha_info(imtrain_json) :
    dfr=pd.DataFrame(columns=['boundingPoly','confidence','importanceFraction','prod'])
    try :
        jchas=json_normalize(imtrain_json.cropHintsAnnotation)
        for k in range(0,len(jchas.iloc[0][0])) :
            imtrain_data=[str(jchas.iloc[0][0][k]['boundingPoly']),
                       jchas.iloc[0][0][k]['confidence'],
                       jchas.iloc[0][0][k]['importanceFraction'],
                       jchas.iloc[0][0][k]['confidence']*jchas.iloc[0][0][k]['importanceFraction']]
            imtrain_row=pd.DataFrame(data=imtrain_data,index=['boundingPoly','confidence','importanceFraction','prod']).transpose()
            dfr=dfr.append(imtrain_row)
        dfr=dfr[dfr['prod']==dfr['prod'].max()]
    except :
        print('Line Skipped in JCHA')
        dfr=pd.DataFrame(data=['',0,0,0], index=['boundingPoly','confidence','importanceFraction','prod']).transpose()
    return dfr
    
###*****************************************************************###

def fill_img_df(imtrain_path,k_from, k_to) :
    imtrain_names=os.listdir(imtrain_path)
    k_to=min(k_to,len(imtrain_names))
    imtrain_names=imtrain_names[k_from:k_to]
    col_names=['PetID','ImgID','jla_description','jla_score','jpa_color','jpa_score','jpa_pixel_fract','jcha_bounding','jcha_confidence','jcha_import_fract']
    img_df = pd.DataFrame(columns=col_names)
    i=k_from
    for imtrain_name in imtrain_names :
        i=i+1
        if (i%1000==0) : print('Current File nr.:{}'.format(i),datetime.datetime.now())
        imtrain_file = imtrain_path + imtrain_name
        pet_id=imtrain_name.split('.')[0].split('-')[0]
        img_id=imtrain_name.split('.')[0].split('-')[1]
        #
        imtrain_json=pd.read_json(imtrain_file, orient='index', typ='series')
        info_jla=extr_jla_info(imtrain_json)  # contains description e score
        info_jpa=extr_jpa_info(imtrain_json)  # contains RGB as string, score, pixelFraction
        info_jcha=extr_jcha_info(imtrain_json) # contains boundingPoly, confidence, importanceFraction
        #
        imtrain_row=pd.DataFrame(data=[pet_id, img_id,
                                    not_empty(info_jla['description']),
                                    not_empty(info_jla['score']),
                                    not_empty(info_jpa['color']),
                                    not_empty(info_jpa['score']),
                                    not_empty(info_jpa['pixelFraction']),
                                    not_empty(info_jcha['boundingPoly']),
                                    not_empty(info_jcha['confidence']),
                                    not_empty(info_jcha['importanceFraction'])],
                 index=col_names).transpose()
        img_df=img_df.append(imtrain_row)
    return img_df
    
###*****************************************************************###
# The process will take a lot of time, make up 3 restore points.

img_df1=fill_img_df('../input/train_metadata/',0,20000)

    
###*****************************************************************###

img_df2=fill_img_df('../input/train_metadata/',20000,40000)
    
###*****************************************************************###

img_df3=fill_img_df('../input/train_metadata/',40000,60000)
    
###*****************************************************************###

img_df=pd.concat([img_df1,img_df2,img_df3], axis=0)
    
###*****************************************************************###

col_names=['jla_score','jpa_score','jpa_pixel_fract','jcha_confidence','jcha_import_fract']
img_df = cols_to_numeric(img_df,col_names)
img_df.dtypes
    
###*****************************************************************###


img_df['rating']=img_df['jla_score']*img_df['jpa_score']*img_df['jpa_pixel_fract']*img_df['jcha_confidence']*img_df['jcha_import_fract']


## dont't know how to use these columns with the others
img_df.drop('jpa_color', axis=1, inplace=True)
img_df.drop('jcha_bounding', axis=1, inplace=True)


img_df.sort_values(['PetID'], inplace=True)


img_df.head(8)


img_dfd=img_df[['PetID','jla_description']].groupby(['PetID']).min()
img_dfd


img_dfm=img_df.groupby(['PetID']).mean()
img_dfm.head()


img_dfc=img_df[['PetID']].groupby(['PetID']).size()
img_dfc.head()


img_fdata=img_dfd


col_names = ['jla_score','jpa_score','jpa_pixel_fract','jcha_confidence','jcha_import_fract', 'rating']
for cn in col_names :
    img_fdata[cn]=img_dfm[cn]


img_fdata['cnt']=img_dfc
img_fdata.head()


img_fdata.drop('jla_description', axis=1, inplace=True)
#img_fdata.drop('ImgID', axis=1, inplace=True)


img_fdata.head()


#img_fdata.to_csv('img_fdata.csv', index=True)


train.info()


train.Age = train.Age.astype(np.float64)
train['Age'] = train['Age'].replace([0.0],train['Age'].mode())



pb = list()

for element in train.Breed2:
    if element == 0:
        pb.append(1)
    else:
        pb.append(0)

train['PureBreed'] = pb
train['PureBreed']


print(train['PureBreed'].value_counts())


#train['FurLength'].value_counts()
#train['FurLength'].describe()

#print(train['PhotoAmt'].value_counts())
#print('*'*50)
#print(train['PhotoAmt'].describe())

#train.groupby('Age').count()

#train.loc[(train['Breed2'] == 0)& (train['Breed1'] == 0)]

# train.loc[train['Breed2'] == 0] 
# train.loc[train['Breed2'] != 0]
# train.loc[train['Breed1'] == 0]

#train.loc[(train['Color1'] == 0)& (train['Color2'] == 0) & (train['Color3'] != 0)]


print(train.info())
print("=================================================================================")
print(txt_train.info())
print("=================================================================================")
print(img_fdata.info())


train = train.set_index('PetID')
txt_train = txt_train.set_index('PetID')
#X = img_fdata.set_index('PetID')


train_all=pd.concat([train,txt_train,img_fdata], axis=1)
train_all


train_all.info()





train_all[['magnitude','score', 'jla_score','jpa_score','jpa_pixel_fract', 'jcha_confidence', 'jcha_import_fract', 'cnt', 'rating']]= train_all[['magnitude','score', 'jla_score','jpa_score','jpa_pixel_fract', 'jcha_confidence', 'jcha_import_fract', 'cnt', 'rating']].fillna(0.0)


train_all['VideoAmt'].tail(10)


train_all['State'].nunique()


X = train_all[['Type', 'Age', 'Breed1', 'Breed2','PureBreed', 'Gender', 'Color1', 'Color2','Color3', 'MaturitySize',
           'FurLength', 'Vaccinated', 'Dewormed','Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'magnitude',
           'score', 'jla_score', 'jpa_score', 'jpa_pixel_fract', 'jcha_confidence', 'jcha_import_fract', 
           'cnt', 'rating']]

Y = train_all[['AdoptionSpeed']]


X.info()


# test.columns


# test.info()


# test.Age = test.Age.astype(np.float64)
# test['Age'] = test['Age'].replace([0.0],test['Age'].mode())
#

# pb = list()

# for element in test.Breed2:
#     if element == 0:
#         pb.append(1)
#     else:
#         pb.append(0)

# test['PureBreed'] = pb

# Xtest = test[['PetID', 'Type', 'Age', 'Breed1', 'Breed2','PureBreed', 'Gender', 'Color1', 'Color2','Color3', 
#      'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 
#      'Fee', 'State', 'VideoAmt', 'PhotoAmt']]

# Xtest = Xtest.set_index('PetID')

# Xtest.columns


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


#Support Vector
svclf = SVC(gamma='scale', decision_function_shape='ovo')
svclf.fit(X_train, np.ravel(Y_train))


predicted = svclf.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.27849636216653195
# 0.28314470493128535


# KNeighborsClassifier
knnclf = KNeighborsClassifier()
knnclf.fit(X_train, np.ravel(Y_train))
predicted = knnclf.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.3286176232821342
# 0.3356911883589329


gnbclf = GaussianNB()

gnbclf.fit(X_train, np.ravel(Y_train))
predicted = gnbclf.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)
#0.31244947453516575
# 0.29203718674211804


dtc = DecisionTreeClassifier()

dtc.fit(X_train, np.ravel(Y_train))
predicted = gnbclf.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.29203718674211804


# GridSearchCV
# svc 
#C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False
# , 'poly', 'sigmoid', 'precomputed'
svclf = SVC()
parameters = {'C':[1, 10]}

gssvc = GridSearchCV(svclf, parameters, cv = 5)

gssvc.fit(X, np.ravel(Y))


gssvc.best_score_

gssvc.best_params_




















rfclf = RandomForestClassifier()

rfclf.fit(X_train, np.ravel(Y_train))
predicted = rfclf.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.3654001616814875
# 0.35246564268391267


#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.preprocessing import MultiLabelBinarizer

xgbclf = XGBClassifier()

xgbclf.fit(X_train, np.ravel(Y_train))
predicted = xgbclf.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.40541632983023446
# 0.39894907033144705


#pred_xgbclf = xgbclf.predict(Xtest)
#print(np.where(pred_xgbclf < 0)[0].shape)

#test['AdoptionSpeed']=pred_xgbclf.astype(int)
#outpred_xgbclf = test[['PetID','AdoptionSpeed']]

#outpred_xgbclf.to_csv('submission.csv',index=False)


# MLPClassifier

mlp = MLPClassifier()

#mlp.fit(X_train, Y_train.values.ravel())
#predicted = mlp.predict(X_test)
#expected = Y_test

#accuracy_score(expected,predicted)

# 0.2627324171382377


parameters = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

mlpgs = GridSearchCV(mlp, parameters, cv=5)
mlpgs.fit(X_train, Y_train.values.ravel())

#print_results(mlpgs)


mlpgs.best_score_

#0.28063713290194126
# 0.27834743653558985


mlpgs.best_params_


mlpgs.best_estimator_


predicted = mlpgs.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.27849636216653195
# 0.28314470493128535

## LogisticRegression

logr = LogisticRegression(C=1, multi_class= 'multinomial', solver= 'newton-cg') 
logr.fit(X_train, np.ravel(Y_train))

# 0.32821341956346

predicted = logr.predict(X_test)
expected = Y_test

accuracy_score(expected,predicted)

#0.3617623282134196 -> C=1, multi_class= 'multinomial', solver= 'newton-cg'

parameters = {'C':[5, 1, 7], 'solver': ['newton-cg', 'lbfgs', 'saga'], 'multi_class': ['multinomial', 'auto']}

gssvc = GridSearchCV(logr, parameters, cv = 9)

gssvc.fit(X, np.ravel(Y))


gssvc.best_score_

gssvc.best_params_

#0.328286533715734
# 0.3570332821983592 -->> {'C': 5, 'solver': 'newton-cg'}
# 0.3560328153138131 -->> {'C': 1, 'multi_class': 'multinomial', 'solver': 'newton-cg'}




######################################################################################################################################################
##**************************************************************************************************************************************************##
######################################################################################################################################################
##**************************************************************************************************************************************************##

imtest_path= '../input/test_metadata/'
imtest_names=os.listdir('../input/test_metadata')
len(imtest_names)


imtest_name=os.listdir('../input/test_metadata')[0]
#imtrain_name
imtest_file = imtest_path + imtest_name
imtest_file

imtest_json=pd.read_json(imtest_file, orient='index', typ='series')
imtest_json

jla=json_normalize(imtest_json.labelAnnotations)
jla

jla['description'].str.cat(sep=', ')

jla[jla['score']==jla['score'].max()]

jpa=json_normalize(imtest_json.imagePropertiesAnnotation)
jpa.iloc[0]

jpa.iloc[0][0]

len(jpa.iloc[0][0])

k=2

jpa.iloc[0][0][k]['color']

jpa.iloc[0][0][k]['score']

jpa.iloc[0][0][k]['pixelFraction']

df_jpa=pd.DataFrame(columns=['color','score','pixelFraction'])
df_jpa

data=[str(jpa.iloc[0][0][k]['color']),jpa.iloc[0][0][k]['score'],jpa.iloc[0][0][k]['pixelFraction']]
pd.DataFrame(data=data,index=['color','score','pixelFraction']).transpose()

df_jpa=pd.DataFrame(columns=['color','score','pixelFraction'])
for k in range(0,len(jpa.iloc[0][0])) :
    curr_data=[str(jpa.iloc[0][0][k]['color']),jpa.iloc[0][0][k]['score'],jpa.iloc[0][0][k]['pixelFraction']]
    curr_row=pd.DataFrame(data=curr_data,index=['color','score','pixelFraction']).transpose()
    df_jpa=df_jpa.append(curr_row)
df_jpa['prod']=df_jpa['score']*df_jpa['pixelFraction']
df_jpa

len(df_jpa[df_jpa['prod']==df_jpa['prod'].max()])

jcha=json_normalize(imtest_json.cropHintsAnnotation)
jcha.iloc[0]

jcha.iloc[0][0]

jcha.iloc[0][0][0]['boundingPoly']

jcha.iloc[0][0][0]['confidence']

jcha.iloc[0][0][0]['importanceFraction']


# Forked: (Pets Adoption Simple (pandas + random forest))
# #  Thought: Is it possible to summarise all this data? 
# #  The idea is to create a dataframe to join later with df_all. As there are many images per pet, when we will join all the data we will have to choose one of the image rows, using a function like max() or sum().
# #  First, some useful secondary functions:
# #


###*****************************************************************###

def not_empty(s) :
    if len(s)>=1 : 
        return s[0]
    else :
        return ''
    
###*****************************************************************###

def extr_jla_info(imtest_json) :
    dfr=pd.DataFrame(columns=['score','description'])
    try :
        jlas=json_normalize(imtest_json.labelAnnotations)
        imtest_data=[jlas[jlas['score']==jlas['score'].max()]['score'][0],
                   jlas['description'].str.cat(sep=',')]
        imtest_row=pd.DataFrame(data=imtest_data,index=['score','description']).transpose()
        dfr=dfr.append(imtest_row)
    except:
        print('Line Skipped in JLA')
        dfr=pd.DataFrame(data=[0,0], index=['score','description']).transpose()
    return dfr
    
###*****************************************************************###

def extr_jpa_info(imtest_json) :
    dfr=pd.DataFrame(columns=['color','score','pixelFraction','prod'])
    try :
        jpas=json_normalize(imtest_json.imagePropertiesAnnotation)
        for k in range(0,len(jpas.iloc[0][0])) :
            imtest_data=[str(jpas.iloc[0][0][k]['color']),
                   jpas.iloc[0][0][k]['score'],
                   jpas.iloc[0][0][k]['pixelFraction'],
                   jpas.iloc[0][0][k]['score']*jpas.iloc[0][0][k]['pixelFraction']]
            imtest_row=pd.DataFrame(data=imtest_data,index=['color','score','pixelFraction','prod']).transpose()
            dfr=dfr.append(imtest_row)
        dfr=dfr[dfr['prod']==dfr['prod'].max()]    
    except :
        print('Line Skipped in JPA')
        dfr=pd.DataFrame(data=['',0,0,0], index=['color','score','pixelFraction','prod']).transpose()
    return dfr
    
###*****************************************************************###

def extr_jcha_info(imtest_json) :
    dfr=pd.DataFrame(columns=['boundingPoly','confidence','importanceFraction','prod'])
    try :
        jchas=json_normalize(imtest_json.cropHintsAnnotation)
        for k in range(0,len(jchas.iloc[0][0])) :
            imtest_data=[str(jchas.iloc[0][0][k]['boundingPoly']),
                       jchas.iloc[0][0][k]['confidence'],
                       jchas.iloc[0][0][k]['importanceFraction'],
                       jchas.iloc[0][0][k]['confidence']*jchas.iloc[0][0][k]['importanceFraction']]
            imtest_row=pd.DataFrame(data=imtest_data,index=['boundingPoly','confidence','importanceFraction','prod']).transpose()
            dfr=dfr.append(imtest_row)
        dfr=dfr[dfr['prod']==dfr['prod'].max()]
    except :
        print('Line Skipped in JCHA')
        dfr=pd.DataFrame(data=['',0,0,0], index=['boundingPoly','confidence','importanceFraction','prod']).transpose()
    return dfr
    
###*****************************************************************###

def fill_img_df(imtest_path,k_from, k_to) :
    imtest_names=os.listdir(imtest_path)
    k_to=min(k_to,len(imtest_names))
    imtest_names=imtest_names[k_from:k_to]
    col_names=['PetID','ImgID','jla_description','jla_score','jpa_color','jpa_score','jpa_pixel_fract','jcha_bounding','jcha_confidence','jcha_import_fract']
    img_df = pd.DataFrame(columns=col_names)
    i=k_from
    for imtest_name in imtest_names :
        i=i+1
        if (i%1000==0) : print('Current File nr.:{}'.format(i),datetime.datetime.now())
        imtest_file = imtest_path + imtest_name
        pet_id=imtest_name.split('.')[0].split('-')[0]
        img_id=imtest_name.split('.')[0].split('-')[1]
        #
        imtest_json=pd.read_json(imtest_file, orient='index', typ='series')
        info_jla=extr_jla_info(imtest_json)  # contains description e score
        info_jpa=extr_jpa_info(imtest_json)  # contains RGB as string, score, pixelFraction
        info_jcha=extr_jcha_info(imtest_json) # contains boundingPoly, confidence, importanceFraction
        #
        imtest_row=pd.DataFrame(data=[pet_id, img_id,
                                    not_empty(info_jla['description']),
                                    not_empty(info_jla['score']),
                                    not_empty(info_jpa['color']),
                                    not_empty(info_jpa['score']),
                                    not_empty(info_jpa['pixelFraction']),
                                    not_empty(info_jcha['boundingPoly']),
                                    not_empty(info_jcha['confidence']),
                                    not_empty(info_jcha['importanceFraction'])],
                 index=col_names).transpose()
        img_df=img_df.append(imtest_row)
    return img_df
    
###*****************************************************************###
# The process will take a lot of time, make up 3 restore points.

timg_df1=fill_img_df('../input/test_metadata/',0,20000)

    
###*****************************************************************###

timg_df2=fill_img_df('../input/test_metadata/',20000,40000)
    
###*****************************************************************###

timg_df3=fill_img_df('../input/test_metadata/',40000,60000)
    
###*****************************************************************###

timg_df=pd.concat([img_df1,img_df2,img_df3], axis=0)
    
###*****************************************************************###

col_names=['jla_score','jpa_score','jpa_pixel_fract','jcha_confidence','jcha_import_fract']
timg_df = cols_to_numeric(timg_df,col_names)
timg_df.dtypes
    
###*****************************************************************###

timg_df['rating']=timg_df['jla_score']*timg_df['jpa_score']*timg_df['jpa_pixel_fract']*timg_df['jcha_confidence']*timg_df['jcha_import_fract']

## dont't know how to use these columns with the others
timg_df.drop('jpa_color', axis=1, inplace=True)
timg_df.drop('jcha_bounding', axis=1, inplace=True)

timg_df.sort_values(['PetID'], inplace=True)

timg_df.head(8)

timg_dfd=timg_df[['PetID','jla_description']].groupby(['PetID']).min()
timg_dfd

timg_dfm=timg_df.groupby(['PetID']).mean()
timg_dfm.head()

timg_dfc=timg_df[['PetID']].groupby(['PetID']).size()
timg_dfc.head()

timg_fdata=timg_dfd

col_names = ['jla_score','jpa_score','jpa_pixel_fract','jcha_confidence','jcha_import_fract', 'rating']
for cn in col_names :
    timg_fdata[cn]=timg_dfm[cn]


timg_fdata['cnt']=timg_dfc
timg_fdata.head()

timg_fdata.drop('jla_description', axis=1, inplace=True)

timg_fdata.head()

#img_fdata.to_csv('img_fdata.csv', index=True)

test.info()

test.Age = test.Age.astype(np.float64)
test['Age'] = test['Age'].replace([0.0],test['Age'].mode())


pb = list()

for element in test.Breed2:
    if element == 0:
        pb.append(1)
    else:
        pb.append(0)

test['PureBreed'] = pb
test['PureBreed']

print(test['PureBreed'].value_counts())

print(test.info())
print("=================================================================================")
print(txt_test.info())
print("=================================================================================")
print(timg_fdata.info())

test = test.set_index('PetID')
txt_test = txt_test.set_index('PetID')
test
test_all=pd.concat([test,txt_test,timg_fdata], axis=1, join = 'inner')
test_all
test_all[['magnitude','score', 'jla_score','jpa_score','jpa_pixel_fract', 'jcha_confidence', 'jcha_import_fract', 'cnt', 'rating']]= test_all[['magnitude','score', 'jla_score','jpa_score','jpa_pixel_fract', 'jcha_confidence', 'jcha_import_fract', 'cnt', 'rating']].fillna(0.0)

test_all['VideoAmt'].tail(10)

test_all['State'].nunique()

Xt = test_all[['Type', 'Age', 'Breed1', 'Breed2','PureBreed', 'Gender', 'Color1', 'Color2','Color3', 'MaturitySize',
           'FurLength', 'Vaccinated', 'Dewormed','Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'magnitude',
           'score', 'jla_score', 'jpa_score', 'jpa_pixel_fract', 'jcha_confidence', 'jcha_import_fract', 
           'cnt', 'rating']]

# if any in data
### train.isnull().values.any()
# total number of NaN values
#### train.isnull().sum().sum()
# find just the columns that have NaN values
#train.isnull().any()
# number of NaN values in a column
# df[df['name column'].isnull()]
Xt.isnull().sum()
Xt
#Y = xgbclf.predict(Xtest)


Yt = logr.predict(Xt)
#print(np.where(pred_xgbclf < 0)[0].shape)

#test['AdoptionSpeed']=pred_xgbclf.astype(int)
#outpred_xgbclf = test[['PetID','AdoptionSpeed']]

#outpred_xgbclf.to_csv('submission.csv',index=False)