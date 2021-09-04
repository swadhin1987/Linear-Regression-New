# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:47:37 2021

@author: Swadhin
"""
Linear Regression
1)Relationship that states how the two variables are Linerly changing.
2)It Helps us to predict an outcomes(Dependent variable)using indipendent variables
3)Equation- Y=Mx+c


Assumptions of Liner Regression
1)There should be Linear Relationship Between DV&IVS
2)The IVS shold be Non correlated
3)Actual-Predicted=Error---Which is call the Resuduals.
Residuals should be Normal Distrubuted.
4)Residual Should not have any Autocorrelation.
5)Sum of Residuals should be close to 0 and average of the Residual should be constant
6)Varriance of Errors should be Constant called as (Homoscadascity)
7)If varriance of Errors Not constant its called as Heteroscadascity.
8)Errors Should not be uto correlated.
9)IVS are non co-rrelated with each other.

Project no-1
Business Gol:A chinese utomobile company Geely uto Aspires to Enter the US Market by seeting up there Manufacturing unit 
there and producing cars locally to give competition to there US & Europe Counterparts

They have contracted an Automobile consulting company to understand the factors on which the pricing of the cars depends.
specially they want to understand the factors ffecting Prices of cars in the US market,Since those may be very differnt
from the chines market the company wants to know.


which variables re significant in predicting the price of the car?
How well those variables describe the price of a car?
Bsed on the various market survey,the counsulting firm has gathered  large dataset of different typesof cars
across the American Market.

Objective:To Predict Price of Car in a New Market.
History Data:Existing US market car's Data is our History Data.
DV:PRICE 
IVS:colour,Manufacturing,Model,HP,Cylinder-----etc.

    Y=Intercept+b1x1+b2x2+b3x3
    
import pandas as pd
car_df=pd.read_csv("C:\\Users\\Swadhin\\Desktop\\Cars_Retail_Price.csv")

###Data Exploration

###Columns name
car_df.dtypes

# data structure:(we know the missing values in this)
car_df.info()

###How to find the outliars
we will take the mean,median,mode of the variable if the mean,median,mode values are

# to summarize data frame
analysis=car_df.describe()
pd.DataFrame(analysis)

# exploring categorical variables:


for table in ['Cylinder', 'Doors', 'Cruise', 'Sound','Leather']:
    car_df[table] = car_df[table].astype('category')
    
    
   
cat_vars = car_df.select_dtypes(include='category').columns.tolist()
cat_vars=list(cat_vars)


for i in cat_vars:
    x=car_df[i].value_counts() 
    print(x)
    

# checking if missing values exist?
    

car_df.isnull().sum()

# check for outliers for extreme values



# creating dummy variables(one hot encoding):

# create dummies for categorical variables:

car_df['Make'].value_counts()
dummies = pd.get_dummies(car_df['Make']).rename(columns=lambda x: 'Make_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies], axis=1)

car_df['Model'].value_counts()
dummies2 = pd.get_dummies(car_df['Model']).rename(columns=lambda x: 'Model_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies2], axis=1)

car_df['Trim'].value_counts()
dummies3 = pd.get_dummies(car_df['Trim']).rename(columns=lambda x: 'Trim_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies3], axis=1)


car_df['Type'].value_counts()
dummies4 = pd.get_dummies(car_df['Type']).rename(columns=lambda x: 'Type_' + str(x))
# bring the dummies back into the original dataset
car_df = pd.concat([car_df, dummies4], axis=1)

####Feature creation completed
    
    
# drop columns that are not required- categorical(4) + dummies(dummy variable trap ones+ lower frequency ones)

 
 #checking frequencies and dropping variables:

 
car_df['Make'].value_counts()  # drop Make_Saturn
car_df['Model'].value_counts()  # keep Model_Malibu, Model_Cavalier, Model_AVEO, Model_Cobalt, Model_Ion
car_df['Trim'].value_counts()   # keep Sedan4D, Coupe 2D , LS Sedan 4D, LS Coupe 2D, LT Sedan 4D 
car_df['Type'].value_counts()   #drop convertible
 
 
 #keep shorlisted vars:
   
new_df=car_df[['Price', 'Mileage','Cylinder','Liter','Doors','Cruise','Sound','Leather','Make_Buickuick','Make_Cadillac','Make_Chevrolet','Make_Pontiac','Make_SAAB','Model_Malibu','Model_Cavalier','Model_AVEO', 'Model_Cobalt',  'Model_Ion','Trim_Sedan 4D','Trim_Coupe 2D','Trim_LS Sedan 4D','Trim_LS Coupe 2D','Trim_LT Sedan 4D', 'Type_Coupe','Type_Hatchback','Type_Sedan','Type_Wagon']]
 
 # removing variables that are near zero/ low standard dev/variance
 
import numpy as np
x=np.var(car_df, axis=0)
x=pd.DataFrame(x)
x.to_csv("nzv_cars.csv")
new_df=car_df[['Mileage',
'Cylinder',
'Liter',
'Doors',
'Make_Chevrolet',
'Type_Sedan',
'Sound',
'Leather',
'Cruise',
'Trim_Sedan 4D',
'Make_Pontiac',
'Type_Coupe',
'Make_Buick',
'Make_SAAB',
'Make_Cadillac',
'Type_Wagon',
'Make_Saturn',
'Model_AVEO',
'Model_Cavalier',
'Model_Malibu',
'Type_Hatchback',
'Model_Cobalt',
'Model_Ion',
'Trim_Coupe 2D',
'Trim_LS Sedan 4D',
'Type_Convertible',
'Model_9_3 HO',
'Price',]]

    
### dropping correlated var ####
###Hear we will see the co-relation Between the Variable 
co-relation stands in -1 to 1
-100% is called as the -ve co-relation=One thing will increase other thing will decrease
100% is called +ve co-relation=one thing will increse symiltaneously other thing also increse
we will take the cutoff list of varible by 60%
like +ve 60%>strong +ve linear relation----strong co-related
-60%<strong -ve linear reltion----strong co-related
fter this cutoff we Reduces 4 variables
 
cor=new_df.corr()
 
cor.to_csv('cor_linear_reg.csv')   
import os
os.getcwd()
 
new_df_model=new_df.drop(['Liter',
'Type_Sedan',
'Model_Ion',
'Type_Coupe',], axis=1)
 
 
 #converting all variables to numeric:
 
model_df = new_df_model.apply(pd.to_numeric)

   

#model iterations:Model Building Phase
in this phase we have the 
Histroy data is having 2 things,1=Train Data,2=Test Data
we are considering 70% is the Train Data and 30% is the Test Data

x = model_df.iloc[:,1:24]
y = model_df.iloc[:,0]


#Splitting the data into training and test sets
Explanation of this Syntex
My Entire History Data=804 car
70% is the Training Data(562 no of Car are present),30% Test Data(232 car are present)
Train Data is also Diveded in 2 group.1=x_train data(562,23(Ivs)),2=y_train data(562,1DV)
Test data is also divided into 2 groups.1=x_test data(232,23 Ivs),2=y_test data (232,1Dv)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,) 


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm = lm.fit(x_train,y_train)

y_pred =lm.predict(x_train)

y_pred=pd.DataFrame(y_pred)
y_train=pd.DataFrame(y_train)
y_test.to_csv('y_test.csv')
y_pred.to_csv('y_pred.csv')

#checking coefficients:

lm.coef_

import numpy as np
coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)

lm.intercept_


#To predict the values of y on the test set we use lm.predict( )

y_pred = lm.predict(x_test)

#Residuals/Errors are the difference between observed and predicted values.
#y_error = y_test - y_pred

y_pred=pd.DataFrame(y_pred)

y_test.to_csv('y_test.csv') 
y_pred.to_csv('y_pred.csv') 
y_error.to_csv('y_error.csv') 
import os
os.getcwd()


#R square can be obbtained using sklearn.metrics ( ):
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)




