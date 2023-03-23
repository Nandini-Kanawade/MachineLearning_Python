#'pandas' library for working with data sheets
import pandas as pd 
#'numpy' library for working with arrays
import numpy as np
#'matplotlib' library for working with data visulization methods (plots)
import matplotlib.pyplot as plt

#Dictionary Creation using {}
d={ 'Student_Name' :pd.Series(['ABC','AQW','NHY','CBY','SPO','LHG','NZR','MLG','AUF','DOX']),
    'Maths':pd.Series([50,84,56,88,63,93,56,65,78,89]),
    'Electronic_Circuits' :pd.Series([70,62,98,95,55,85,45,59,43,67]),
    'Signals_Systems'     :pd.Series([64,75,45,76,56,71,75,45,76,62]),
    'Analog_Communication':pd.Series([43,46,61,84,84,65,46,61,84,75]),
    'Digital_Electronics' :pd.Series([51,74,73,79,95,58,74,73,79,46]),
   }
   
#Dictionary to dataframe
df = pd.DataFrame(d)
print(df) 

#Mathematics Line Chart
plt.plot(df['Student_Name'],df['Maths'])
plt.title('Mathematics')
plt.xlabel('Student Name')
plt.ylabel('Mathematics Marks')
plt.grid()
plt.show()

#Electronic Circuits Line Chart
plt.plot(df['Student_Name'],df['Electronic_Circuits'])
plt.title('Electronic Circuits')
plt.xlabel('Student Name')
plt.ylabel('Electronic Circuits Marks')
plt.grid()
plt.show()

#Signals Systems Line Chart
plt.plot(df['Student_Name'],df['Signals_Systems'])
plt.title('Signals and Systems')
plt.xlabel('Student Name')
plt.ylabel('Signals Systems Marks')
plt.grid()
plt.show()

#Analog Communication Line Chart
plt.plot(df['Student_Name'],df['Analog_Communication'])
plt.title('Analog Communication')
plt.xlabel('Student Name')
plt.ylabel('Analog Communication Marks')
plt.grid()
plt.show()

#Digital Electronics Line Chart
plt.plot(df['Student_Name'],df['Digital_Electronics'])
plt.title('Digital Electronics')
plt.xlabel('Student Name')
plt.ylabel('Digital Electronics Marks')
plt.grid()
plt.show()

#Bar Charts of Subject Vs Name of Student
#The grid here is not needed as the top of the bars are well-defined.

#Mathematics Bar Chart
plt.bar(df['Student_Name'],df['Maths'])
plt.title('Mathematics')
plt.xlabel('Student Name')
plt.ylabel('Mathematics Marks')
plt.show()

#Electronic Circuits Bar Chart
plt.bar(df['Student_Name'],df['Electronic_Circuits'])
plt.title('Electronic Circuits')
plt.xlabel('Student Name')
plt.ylabel('Electronic Circuits Marks')
plt.show()

#Signals Systems Bar Chart
plt.bar(df['Student_Name'],df['Signals_Systems'])
plt.title('Signals and Systems')
plt.xlabel('Student Name')
plt.ylabel('Signals Systems Marks')
plt.show()

#Analog Communication Bar Chart
plt.bar(df['Student_Name'],df['Analog_Communication'])
plt.title('Analog Communication')
plt.xlabel('Student Name')
plt.ylabel('Analog Communication Marks')
plt.show()

#Digital Electronics Bar Chart
plt.bar(df['Student_Name'],df['Digital_Electronics'])
plt.title('Digital Electronics')
plt.xlabel('Student Name')
plt.ylabel('Digital Electronics Marks')
plt.show()

#Histogram of Marks Obtained in a Subject vs Number of Students.
#For differentiating the individual blocks we have bordered the 
#bars with black color and using '.xticks()' we defined the full range
#of marks from 0-100 and hence in onelook of Histogram we can better understand the data.

#Mathematics Histogram
plt.hist(df['Maths'],[10,20,30,40,50,60,70,80,90,100],ec='black')
plt.title('Mathematics')
plt.xlabel('Mathematics Marks')
plt.ylabel('Number of Students')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.show()

#Electronic Circuits Histogram
plt.hist(df['Electronic_Circuits'],[10,20,30,40,50,60,70,80,90,100],ec='black')
plt.title('Electronic Circuits')
plt.xlabel('Electronic Circuits Marks')
plt.ylabel('Number of Students')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.show()

#Signals & Systems Histogram
plt.hist(df['Signals_Systems'],[10,20,30,40,50,60,70,80,90,100],ec='black')
plt.title('Signals & Systems')
plt.xlabel('Signals & Systems Marks')
plt.ylabel('Number of Students')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.show()

#Analog Communication Histogram
plt.hist(df['Analog_Communication'],[10,20,30,40,50,60,70,80,90,100],ec='black')
plt.title('Analog Communication')
plt.xlabel('Analog Communication Marks')
plt.ylabel('Number of Students')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.show()

#Digital Electronics Histogram
plt.hist(df['Digital_Electronics'],[10,20,30,40,50,60,70,80,90,100],ec='black')
plt.title('Digital Electronics')
plt.xlabel('Digital Electronics Marks')
plt.ylabel('Number of Students')
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.show()

#Scatter Plot of Subject vs Name of Students.
#Here we have used only horizontal gridlines as we can 
#see that the plots are very well is differentiable vertically.

#Mathematics Scatter Plot
plt.scatter(df['Student_Name'],df['Maths'],ec='black')
plt.title('Mathematics')
plt.xlabel('Number of Students')
plt.ylabel('Mathematics Marks')
plt.grid(axis='y')
plt.show()

#Electronic_Circuits Scatter Plot
plt.scatter(df['Student_Name'],df['Electronic_Circuits'],ec='black')
plt.title('Electronic Circuits')
plt.xlabel('Number of Students')
plt.ylabel('Electronic Circuits Marks')
plt.grid(axis='y')
plt.show()

#Signals & Systems Scatter Plot
plt.scatter(df['Student_Name'],df['Signals_Systems'],ec='black')
plt.title('Signals & Systems')
plt.xlabel('Number of Students')
plt.ylabel('Signals & Systems Marks')
plt.grid(axis='y')
plt.show()

#Analog Communication Scatter Plot
plt.scatter(df['Student_Name'],df['Analog_Communication'],ec='black')
plt.title('Analog Communication')
plt.xlabel('Number of Students')
plt.ylabel('Analog Communication Marks')
plt.grid(axis='y')
plt.show()

#Digital Electronics Scatter Plot
plt.scatter(df['Student_Name'],df['Digital_Electronics'],ec='black')
plt.title('Digital Electronics')
plt.xlabel('Number of Students')
plt.ylabel('Digital Electronics Marks')
plt.grid(axis='y')
plt.show()

#Now we have a pieChart showing the marks obtained by
#different students in a particular subject.
#As we see the plot we have used different parameters like Shadow,
#Explode to enhace the plots. Using Explode we can easily signify one
#students marks (see SS plot).

#Mathematics Scatter Plot
plt.pie(df['Maths'], labels=df['Student_Name'])
plt.title('Mathematics')
plt.show()

#Electronic Circuits Scatter Plot
plt.pie(df['Electronic_Circuits'], labels=df['Student_Name'], shadow='True')
plt.title('Electronic Circuits')
plt.show()

#Signals & Systems Scatter Plot
plt.pie(df['Signals_Systems'], labels=df['Student_Name'], shadow='True',explode=[0.1,0,0,0,0,0,0,0,0,0])
plt.title('Signals & Systems')
plt.show()

#Analog Communication Scatter Plot
plt.pie(df['Analog_Communication'], labels=df['Student_Name'],explode=[0.1,0.1,0.1,0,0,0,0,0,0,0], shadow='True',)
plt.title('Analog Communication')
plt.show()

#Digital Electronics Scatter Plot
plt.pie(df['Digital_Electronics'], labels=df['Student_Name'],explode=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
plt.title('Digital Electronics')
plt.show()

#Summary Statistics for each Each Student and Each Course Mean, 
#Median, Mode, Standard deviation, Variance
#NOTE: At places needed we have used formating for 
# reducing the Statistical output 2 decimal places.

#Now we need to import 'scipy' Library for the Statistical Calculations
from scipy import stats

# ARITHMETIC MEAN

print("Subject Wise Arithmetic Mean")
M=np.mean(df['Maths'])
EC=np.mean(df['Electronic_Circuits'])
SS=np.mean(df['Signals_Systems'])
AC=np.mean(df['Analog_Communication'])
DE=np.mean(df['Digital_Electronics'])
print("Mathematics:=", M)
print("Electronic Circuits:=", EC)
print("Signals & Systems:=", SS)
print("Analog Communication:=", AC)
print("Digital Electronics:=", DE)


#GEOMETRIC MEAN

print("Subject Wise Geometric Mean")
M_g=stats.gmean(df['Maths'])
EC_g=stats.gmean(df['Electronic_Circuits'])
SS_g=stats.gmean(df['Signals_Systems'])
AC_g=stats.gmean(df['Analog_Communication'])
DE_g=stats.gmean(df['Digital_Electronics'])
print("Mathematics:={:.2f}".format(M_g))
print("Electronic Circuits:={:.2f}".format(EC_g))
print("Signals & Systems:={:.2f}".format(SS_g))
print("Analog Communication:={:.2f}".format(AC_g))
print("Digital Electronics:={:.2f}".format(DE_g))

#HARMONIC MEAN

print("Subject Wise Harmonic Mean")
M_h=stats.hmean(df['Maths'])
EC_h=stats.hmean(df['Electronic_Circuits'])
SS_h=stats.hmean(df['Signals_Systems'])
AC_h=stats.hmean(df['Analog_Communication'])
DE_h=stats.hmean(df['Digital_Electronics'])
print("Mathematics:={:.2f}".format(M_h))
print("Electronic Circuits:={:.2f}".format(EC_h))
print("Signals & Systems:={:.2f}".format(SS_h))
print("Analog Communication:={:.2f}".format(AC_h))
print("Digital Electronics:={:.2f}".format(DE_h))


#STANDARD DEVIATION

print("Subject Wise Standard Deviation")
M_sd=np.std(df['Maths'])
EC_sd=np.std(df['Electronic_Circuits'])
SS_sd=np.std(df['Signals_Systems'])
AC_sd=np.std(df['Analog_Communication'])
DE_sd=np.std(df['Digital_Electronics'])
print("Mathematics:={:.2f}".format(M_sd))
print("Electronic Circuits:={:.2f}".format(EC_sd))
print("Signals & Systems:={:.2f}".format(SS_sd))
print("Analog Communication:={:.2f}".format(AC_sd))
print("Digital Electronics:={:.2f}".format(DE_sd))


#VARIANCE

print("Subject Wise Variance")
M_v=np.var(df['Maths'])
EC_v=np.var(df['Electronic_Circuits'])
SS_v=np.var(df['Signals_Systems'])
AC_v=np.var(df['Analog_Communication'])
DE_v=np.var(df['Digital_Electronics'])
print("Mathematics:=", M_v)
print("Electronic Circuits:=", EC_v)
print("Signals & Systems:=", SS_v)
print("Analog Communication:= {:.2f}".format(AC_v))
print("Digital Electronics:={:.2f}".format(DE_v))

#MEDIAN

print("Subject Wise Median")
M_med=np.median(df['Maths'])
EC_med=np.median(df['Electronic_Circuits'])
SS_med=np.median(df['Signals_Systems'])
AC_med=np.median(df['Analog_Communication'])
DE_med=np.median(df['Digital_Electronics'])
print("Mathematics:=",M_med)
print("Electronic Circuits:=",EC_med)
print("Signals & Systems:=",SS_med)
print("Analog Communication:=",AC_med)
print("Digital Electronics:=",DE_med)



#MODE, here the value in mode=array[] is the entry with freqency 
#and count=array[] is the number of times it occured

print("Subject Wise Mode")
M_mode=stats.mode(df['Maths'])
EC_mode=stats.mode(df['Electronic_Circuits'])
SS_mode=stats.mode(df['Signals_Systems'])
AC_mode=stats.mode(df['Analog_Communication'])
DE_mode=stats.mode(df['Digital_Electronics'])
print("Mathematics:=",M_mode)
print("Electronic Circuits:=",EC_mode)
print("Signals & Systems:=",SS_mode)
print("Analog Communication:=",AC_mode)
print("Digital Electronics:=",DE_mode)

