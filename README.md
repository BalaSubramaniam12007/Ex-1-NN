<H3>ENTER YOUR NAME : BALASUBRAMANIAM L</H3>
<H3>ENTER YOUR REGISTER NO. : 212224240020</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 02/02/2026</H3>
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
```
#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = pd.read_csv("Churn_Modelling.csv")
print("First 5 rows:\n", data.head())

print("\nDataset Info:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nStatistical Summary:\n")
print(data.describe())

# RowNumber, CustomerId, and Surname don't help prediction
data = data.drop(['RowNumber','CustomerId','Surname'], axis=1)

label = LabelEncoder()
data['Geography'] = label.fit_transform(data['Geography'])
data['Gender'] = label.fit_transform(data['Gender'])

print("\nAfter Encoding:\n", data.head())

X = data.drop('Exited', axis=1).values   # features
y = data['Exited'].values                # target

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("\nNormalized Features (first 5 rows):\n", X_scaled[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)
```

## OUTPUT:
<img width="700" height="450" alt="Screenshot 2026-02-02 174751" src="https://github.com/user-attachments/assets/8c48dfdb-aca6-4656-98ff-ae838288288d" />
<br><br>
<img width="450" height="500" alt="Screenshot 2026-02-02 174801" src="https://github.com/user-attachments/assets/91026444-93e4-4f31-ae7d-505eb8d2b848" />
<br><br>
<img width="250" height="350" alt="Screenshot 2026-02-02 174810" src="https://github.com/user-attachments/assets/0646978f-f216-49ec-b473-71792a2374fc" />
<br><br>
<img width="750" height="650" alt="Screenshot 2026-02-02 174830" src="https://github.com/user-attachments/assets/eb18fe8f-3331-4140-bb8b-5986d20cabc3" />
<br><br>
<img width="700" height="500" alt="Screenshot 2026-02-02 174841" src="https://github.com/user-attachments/assets/57b6b33a-9780-4f4e-b46c-1d8ab0972d0b" />
<br><br>
<img width="250" height="60" alt="Screenshot 2026-02-02 174847" src="https://github.com/user-attachments/assets/78a979b9-2d00-4c1b-a368-d9befad99a60" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


