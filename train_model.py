#import necessary libraries
import numpy as np
import pandas as pd

#load the dataset
df=pd.read_csv('Data/student_data.csv')
print(df.head())

#data processing
#1.drop the student id column since it wont be used
df=df.drop("Student_ID",axis=1)

#2.encode male and female to 0 and 1
df["Gender"]=df["Gender"].map({"Male":0,"Female":1})

#3.encode departments
df["Department"] = df["Department"].astype('category').cat.codes

#Scaling numbers
col_to_scale=["Fee_Balance_KES", "Attendance_Pct", "KCSE_Points"]
for col in col_to_scale:
  mean_val=df[col].mean()
  std_val=df[col].std()
  df[col]=(df[col]-mean_val)/std_val



#split X and Y
#y=dropout column(the answer)
#x=all other columns
x=df.drop("Dropout",axis=1).values
y=df["Dropout"].values

x=x.astype(np.float64)
y=y.astype('int')

print("X shape:",x.shape)
print("Y Shape:",y.shape)


