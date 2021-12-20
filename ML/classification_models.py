import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import shap
## classifiers ##
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
##  ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#### Data filtering by criteria for Cachexia
df_b = pd.read_excel(r'', sheet_name=0) # WRITE PATH TO DATAFILE1 HERE
df = pd.read_excel(r'', sheet_name=0) # WRITE PATH TO DATAFILE2 HERE
filter_bmi = df[(df['BMI Base Line-Weight'].notnull()) & (df['BMI Minimum-Weight'].notnull()) & (df['BMI Base Line-Measurement date']!= df['BMI Minimum-Measurement date'])] ## filter for patients who has 2 weight measurements that are not the same!
weightloss = (filter_bmi['BMI Minimum-Weight'] / filter_bmi ['BMI Base Line-Weight'])*100
cachh = filter_bmi[(filter_bmi['BMI Minimum-Weight'] / filter_bmi ['BMI Base Line-Weight']<=0.98) & (filter_bmi['BMI Base Line-BMI']<=20) | ((filter_bmi['BMI Minimum-Weight'] / filter_bmi ['BMI Base Line-Weight'])<=0.98) & (filter_bmi['BMI Minimum-BMI']<=20)]
cachexia = weightloss[weightloss <= 95]
no_cachexia = weightloss[weightloss > 95]
cachexia_data = df.loc[cachexia.index]
cachexia_data_sixmonths = cachexia_data['BMI Minimum-Measurement date']-cachexia_data['BMI Base Line-Measurement date']
cachexia_days = cachexia_data_sixmonths.dt.days
cachexia_days_six_months = cachexia_days[cachexia_days <= 190]
cachexia_criteria = cachexia_data.loc[cachexia_days_six_months.index] ## first criteria >5% weight loss in 6 months
ind = cachh.index
cachexia_joined_indices = set(list(cachh.index.values) + list(cachexia_criteria.index.values))
cachexia_two_criterias = filter_bmi.loc[cachexia_joined_indices,:]
no_cachexia_criteria = filter_bmi.loc[[x for x in filter_bmi.index if x not in cachexia_two_criterias.index]] ## filtering the dataset for non-cachectic paitents

data_no = df_b.loc[filter_bmi.index,:]
data = data_no.filter(regex='Base Line', axis=1)

data.loc[:,'y'] = 0
data.loc[cachexia_two_criterias.index,'y'] = 1 ## data and filter bmi has the same indexes
data = data.rename(columns={'Panel Liver Base Line-(GGT)':'GGT','Panel Liver Base Line-ALT (GPT)':'GPT','Panel Liver Base Line-AST (GOT)':'GOT','Panel Liver Base Line-Albumin':'Albumin','Panel Liver Base Line-Alkaline Phosphatase':'ALP','Panel Liver Base Line-Bilirubin_ direct':'Bilirubin direct','Panel Liver Base Line-Bilirubin_ total':'Bilirubin total','Panel Liver Base Line-Protein_ total':'Protein total','INR Base Line-Result numeric':'INR','PTT Base Line-Result numeric':'PTT','Creatinine Base Line-Result numeric':'Creatinine','BUN Base Line-Result numeric':'BUN'})
data = data.dropna()

###
X = data.loc[:, data.columns != 'y'].reset_index()
y = data.loc[:, data.columns == 'y']
y = y.reset_index()
y = y.y
print(str(len(X)))

### Oversampling of training data & data split
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('X_train' + str(len(X_train)))
print('X_test' + str(len(X_test)))

columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

X = os_data_X
y = os_data_y['y']

#### testing features
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
X_scaled = X_scaled.loc[:, X_scaled.columns != 'y']
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
X_test_scaled = X_test_scaled.loc[:, X_test_scaled.columns != 'y']

names = ["Nearest Neighbors", "SVC SVM","Decision Tree", "Random Forest", "Naive Bayes","Logistic Regression","XG-Boost"]

classifiers = [
    KNeighborsClassifier(3,weights = 'distance'),
    SVC(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),LogisticRegression(),XGBClassifier()]

#pdf = matplotlib.backends.backend_pdf.PdfPages("mean roc.pdf")
accuracy_score = metrics.accuracy_score
estimate = pd.DataFrame()

for name, clf in zip(names, classifiers):
    clf.fit(X_scaled, y)
    clf.predict(X_test_scaled)
    roc_auc = roc_auc_score(y_test, clf.predict(X_test_scaled))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(X_test_scaled)
    else:
        Z = clf.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, Z)
    for t in range(0, len(thresholds)):
        estimate.loc[t, str(name) + 'threshold'] = thresholds[t]
        if hasattr(clf, "decision_function"):
            estimate.loc[t, str(name) + 'accuracy'] = '%.2f' % (accuracy_score(y_test, list(np.array(clf.decision_function(X_test_scaled) > thresholds[t]) + 0)))
        else:
            estimate.loc[t, str(name) + 'accuracy'] = '%.2f' % (accuracy_score(y_test, list(np.array(clf.predict_proba(X_test_scaled)[:, 1] > thresholds[t]) + 0)))
    figure = plt.figure()
    plt.plot(fpr, tpr, label= str(name) + '(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
