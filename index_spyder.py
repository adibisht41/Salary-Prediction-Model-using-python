# Model to predict salary based on the cabdidate's personal information (dataset is in train_csv.csv)#
#it has been divided into steps from ((1) to (26))
#the commands for certain outputs have been commented. It is advised to remove comments for every step
#to view and analyse the output
###############################################################################
################# Section 1- Data Analysis#####################################
# (1) SALARY PREIDICTION MODEL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline   ->use it in Jupyter Noteboook only

#(2)
df = pd.read_csv('train_csv.csv')
df.reset_index(drop=True, inplace=True)
#print(df.head(5))

#(3) Setting the data which was -1 as null object.
for col in df.columns:
    df.loc[df[col] == -1, col] = np.nan
#print(df.head(5))

#print(df.info)

#(4) For now I am going to ignore Computer Programming, Electrical and etc scores as they are any reflected in the domain scores.
## Also removing DOJ etc.
## Removing ID
## Removing college city and job city for basic model.
basic_cols = [u'Salary', u'Gender', u'10percentage', u'10board', 
          u'12graduation', u'12percentage', u'12board', u'CollegeID', 
          u'CollegeTier', u'Degree', u'Specialization', u'collegeGPA', u'CollegeCityTier', u'CollegeState', 
          u'GraduationYear', u'English', u'Logical', u'Quant', u'Domain', u'conscientiousness', 
          u'agreeableness', u'extraversion', u'nueroticism', u'openess_to_experience', 'Designation']
          
data = df[basic_cols]
## Analysis of the numerical columns.
#print(data.describe())

#(5)
## Lets see the top 5 values in each columns - this will help get a feel of the non-numeric columns too like city etc.
## Which are are non-numerical columns? All columns minus numerical columns
non_num_cols = list(set(data.columns) - set(data._get_numeric_data().columns))

#for col in non_num_cols:
    #print col,'has unique values:' , data[col].unique().shape[0]
    #print data[col].value_counts().iloc[:5]
    #print '\n'

#(6) Cleaning CGPA 
#print(data.collegeGPA.describe())

# Printing histogram according to the cgpa of the college
#data.collegeGPA.hist(bins=10)
#plt.title('College GPA - Histogram')    

data.loc[data.collegeGPA <=10, 'collegeGPA'] = data.collegeGPA*10
#data.collegeGPA.hist?
#plt.xlim(0,100)
#plt.title('College GPA - Histogram')

#(7 )So we want to predict Salary
## But, how is the salary distibuted

#data.Salary.hist(xrot =90, bins = 100)
#plt.title('Salary - Histogram')

# CONCEPT OF LONGTAIL IN HISTOGRAM

#(8) How does 12 board look like?
# data['12board'].value_counts().plot(kind='bar', figsize=(16,5))
## 12 board follows a long tail

## How does 12 board look like for top 30 values?
#data['12board'].value_counts()[:30].plot(kind='bar', figsize=(16,5))
## You can see the main primary boards have peaks and then there is a long tail of boards which are not always unique.
## eg. up, upboard, up board and uttar pradesh board all mean the same thing but are represented differently.
## We could have cleaned this data using some scripts and added this column to our analysis but for now we will not use this.
    
#(9) How do English scores of candidates look?
#data.English.hist(bins=20)
#plt.title('English - Histogram')


#(10) But how does English affect Salary?
## What scores of English get what kinds of Salaries?
## Do higher english scores get you higher salaries?
## Is it after any particular score, that it starts to stop mattering?
#plt.scatter(data.English, data.Salary)
#plt.xlim((100,1000))
#plt.title('English v/s Salary')
#plt.xlabel('English')
#plt.ylabel('Salary') 

#(11) Similarly for Logical Ability
#plt.scatter(data.Logical, data.Salary)
#plt.xlim((100,1000))
#plt.title('Logical Ability vs Salary')
#plt.xlabel('Logical Ability')
#plt.ylabel('Salary')
   
#(12) Gender role  HISTOGRAM + BOXPLOT
#data.boxplot(by='Gender', column='Salary')
#plt.ylim(0,800000)
#data.hist(by='Gender', column='Salary')
## Note read line in median.
## How to read this plot?



###########################################################################
###################Section 2- Making our model#############################

#(13)  we start with Linear Regression
from sklearn.linear_model import LinearRegression

#(14) We are interested in predicting Salary
data.columns
# For now, we will try to predict using only these columns:
X_col = [u'Gender', u'10percentage', u'12graduation', u'12percentage', u'CollegeTier',
     u'Degree', u'collegeGPA', u'CollegeCityTier', u'GraduationYear', u'English', u'Logical', 
     u'Quant', u'Domain', u'conscientiousness', u'agreeableness', u'extraversion', u'nueroticism', u'openess_to_experience']
y_col = 'Salary'
df_exp = df[X_col + [y_col]]
#print(df_exp.info())

#(14) Domain has a few missing values, many models can't handle missing values. For the sake of similicty we will use frame with
## no missing values. We may lose some rows (the ones that have missing values)
df_exp.dropna(subset=X_col, inplace=True) #very important expression

##  We lost a few rows, but it's okay. We could have let go of the domain column too, but we believe it could be important
## in predicting the salary
#print(df_exp.shape)

#(15)
X = df_exp[X_col]
y = df_exp.Salary

#(16)
# How does one row of X look?
#print(X.loc[0])

#(17)
# Not all columns have numerical values. Not all models can handle categorical variables directly. For example, LinearRegression,
# requires completely numerical input. So we will need to convert the categorical varibles to some numerical values.
# Gender and Degree need our attention.

# For GENDER we will convert m --> 1 and f to --> 0 
from sklearn.preprocessing import LabelEncoder
X.loc[X.Gender == 'm', 'Gender'] = 1
X.loc[X.Gender == 'f', 'Gender'] = 0
#print(X.Gender.unique())

# For DEGREE, which has 4 unique values, we will get 4 new columns representing each of the types of Degrees. If a candidate is
# from any one of the degrees then that columns will have the value 1 (like ON) and the rest columns will be 0 (OFF).
#print('Old shape', X.shape)
X = pd.get_dummies(X, columns=['Degree'])
#print('New shape', X.shape)

## Check the last four new columns
#print(X.columns)

# How does one row of X look now?
#print(X.loc[0])

#df_exp = df_exp.replace(np.nan,0)
#df_exp=df_exp.replace(df_exp['Salary'], (df_exp['Salary'])/1000000)  ##not working
#print(df_exp)

#(18) FINALLY TRAINED THE MODEL FOR LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
mdl = LinearRegression()
mdl = mdl.fit(X, y)
## thats it, you've trained your linear regression

##(19) But how does it do? 
explained_variance = mdl.score(X, y)
#print('explained variance =')
#print(explained_variance)
## Note we are predicting the model on the data it  used to learn in the first place.

## What score --> 0.15 mean? In sklearn regression models, scores are r^2 values. Which define the percentage of variange 
## explained. Root of the same number gives r, which is the correlation between the predicted and the observed values.
correlation = np.sqrt(explained_variance)
#print('correlation = %f'%correlation)
#print(correlation)

##(20) What are some other metrics for model eval?
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y, mdl.predict(X))
#print 'Mean squared error =', mse
#print 'Root Mean squared error =', np.sqrt(mse)
#print 'Mean Absolute error =', mean_absolute_error(y, mdl.predict(X))
## These give you a better intuitive sense of how off you are on an average in every prediction.

#(21) What exactly is this corerlation about?
#predicted_values = mdl.predict(X)
#plt.scatter(predicted_values, y)
#plt.title('Predicted v/s Observed')
#plt.plot([0,1000000],[0,1000000], c = 'black', alpha =0.5)

#(22) A little zoomed in?
#plt.scatter(predicted_values, y)
#plt.title('Predicted v/s Observed')
#plt.plot([0,1000000],[0,1000000], c = 'black', alpha =0.5)
#plt.xlim((0,1000000))
#plt.ylim((0,1000000))

#(23) Let's use a decision tree. using DECISION TREE
from sklearn.tree import DecisionTreeRegressor
## Note how each model in sklearn uses the same api ie. fit() and predict()
mdl = DecisionTreeRegressor()
mdl = mdl.fit(X, y)
#print 'score = ',mdl.score(X, y)

predicted_values = mdl.predict(X)
#plt.scatter(predicted_values, y)
#plt.title('Predicted v/s Observed')

#plt.plot([0,4000000],[0,4000000], c = 'black', alpha=0.5)

# What? So the r is 1 !!
# The decision tree was able to predict almost every possible values. Imagine a tree with
#leaves for each data row. That's great! We have a model which predict salary with amazing accuracy. 
#But wait, we just trained the model on that data, it ought to know how to predict. But linearregression didn't give an amazing
#accuracy even on the same data. 
# Our DT, had the flexiblity to train meld itself exactly as per the training samples, while the linear regression has the
#constraint to be a line and so it did the best it could do as a line.
#Nonetheless, we trained our model completely to the dataset. But our motive is to predict the grade for samples the model 
#doesn't know about, but does guesses/predicts based on his past learnings.
# So what we need to do is test our models on unseen data. We will split our current dataset into 2 parts - train and test.


#print 'Current dataset size', X.shape

#(24) Let's sploit it into 2 datasets  SPLIT
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Get it? What happened? We made 4 sets.
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

#(25) lets train the dt on train set. TRAIN FOR DT
#we train on the train set which is 3001 samples.
print('DECISION TREE')
mdl = DecisionTreeRegressor()
mdl = mdl.fit(X_train, y_train)
r2 = mdl.score(X_test, y_test)
print 'On training data', mdl.score(X_train, y_train)
print 'On unseen test data', r2
# print np.sqrt(r2)    
    
# (26)See. That's how bad your model does. That's what you get for using a complicated/overfit model on train and expecting it
# to work better on test. This was toh extreme example of how bad it does. Let's run a linear regression again. 
print('LINEAR REGRESSION')
mdl = LinearRegression()
mdl = mdl.fit(X_train, y_train)
r2 = mdl.score(X_test, y_test)
print 'On training data', mdl.score(X_train, y_train)
print 'On unseen test data', r2

## What are some other metrics for model eval?

mse = mean_squared_error(y_test, mdl.predict(X_test))
print 'Mean squared error is', mse
print 'Root Mean squared error is', np.sqrt(mse)
print 'Mean Absolute error is', mean_absolute_error(y_test, mdl.predict(X_test))
    
    
    
    
    