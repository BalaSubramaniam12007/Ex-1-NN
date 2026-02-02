<H3>ENTER YOUR NAME : BALASUBRAMANIAM L</H3>
<H3>ENTER YOUR REGISTER NO. : 212224240020</H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
TYPE YOUR CODE HERE
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Read the dataset from drive
df = pd.read_csv("Churn_Modelling.csv")
df.head()
#Finding Missing Values
df.isnull().sum()
#Check for Duplicates
df.duplicated()
#Detect Outliers
num_cols = df.select_dtypes(include='number')
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1
outliers = (num_cols < (Q1 - 1.5 * IQR)) | (num_cols > (Q3 + 1.5 * IQR))
outliers.sum()
#split the dataset into input and output
X = df[['CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].copy()

y = df['Exited']

X.loc[:, 'Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

X = pd.get_dummies(X, columns=['Geography'], drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)
#splitting the data for training & Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
#Print the training data and testing data
print("\nFinal Training set shape:", X_train.shape)
print("Final Testing set shape:", X_test.shape)
```
## OUTPUT:
SHOW YOUR OUTPUT HERE
<img width="1780" height="293" alt="image" src="https://github.com/user-attachments/assets/2080652d-d30a-4333-842b-e7cc8ae4f0b8" />


<img width="298" height="461" alt="image" src="https://github.com/user-attachments/assets/d212e58d-c161-4987-a296-9b77798cc290" />

<img width="327" height="373" alt="image" src="https://github.com/user-attachments/assets/98dce30f-4b2c-4f3c-bfcf-84bb229a5b2b" />

<img width="314" height="372" alt="image" src="https://github.com/user-attachments/assets/4457bb79-c0a5-4841-8b14-f2aa732c9082" />

<img width="444" height="87" alt="image" src="https://github.com/user-attachments/assets/332d67b4-10e8-4b46-ba2a-262c0dd2a2f1" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



