###########      IMPORT LIBRAIRIES      ############

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='white')

st.write("""
# Diabetes Detection
""")

#Open and display an image
image = Image.open('diabetes_img_1.jpg')
st.image(image, caption='TEST', use_column_width=True)


############      LOADING DATASET      ############

df = pd.read_csv('diabetes_2.csv')
st.subheader('Data Information:')

# Some features in the dataset have a value of 0, which denotes missing data. --> replace 0 for NaN
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


############      REPLACING MISSING VALUES    ############

# glucose, blood pressure, skin thickness, insulin, and BMI, all have missing values. Use the ‘Outcome’ variable to find the mean to replace missing data

# Function to find the mean to replace missing data
@st.cache(persist= True)
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
    return temp

#Glucose
median_target("Glucose")
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3

#Blood Pressure
median_target("BloodPressure")
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3

#Skin Thickness
median_target("SkinThickness")
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0

#Insulin
median_target("Insulin")
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8

#Bmi
median_target("BMI")
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4

#data as a table
st.write(df.describe())

############      MODEL BUILDING      ############

# Function to Split the data into independent 'X' and dependent 'y' variables (predictor and target variables)
@st.cache(persist=True)
def split(df):
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    # On met tout sur la même échelle
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    # Split the dataset into 80% Training set and 20% Testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split(df)

#create and train the model
GradientBoostingClassifier = GradientBoostingClassifier()
GradientBoostingClassifier.fit(X_train, y_train)
#X = df.iloc[:, 0:8].values
#Y= df.iloc[:,-1].values
# X_col = df.iloc[:, 0:8]
# Y_col = df.iloc[:,-1]

#Function for the input from the user
def get_user_input():
    pregnancies = st.sidebar.number_input('pregnancies', min_value=0, max_value=None, value=0, step=1)
    glucose = st.sidebar.number_input('glucose', min_value=0.00, max_value=None, value=121.69, step=0.01)
    blood_pressure = st.sidebar.number_input('blood_pressure', min_value=0.00, max_value=None, value=72.42, step=0.01) 
    skin_thickness = st.sidebar.number_input('skin_thickness', min_value=0.00, max_value=None, value=29.24, step=0.01)
    insulin = st.sidebar.number_input('insulin', min_value=0.00, max_value=None, value=159.99, step=0.01)
    BMI = st.sidebar.number_input('BMI', min_value=0.00, max_value=None, value=32.44, step=0.01)
    DPF = st.sidebar.number_input('DPF', min_value=0.00, max_value=None, value=0.47, step=0.01)
    age = st.sidebar.number_input('age', min_value=0, max_value=None, value=33, step=1)
    
    #store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }
    
    #transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the user input into a variable  
user_input = get_user_input()

#set a subheader and diplay the user input
st.subheader('User Input :')
st.write(user_input)


############################      DATA ANALYTICS - BoxPlots     ################################

###   PREGNANCIES   ###
fig1, ax1 = plt.subplots(figsize=(10,2))
sns.boxplot(df['Pregnancies'], color="plum", width=.5)
sns.stripplot(data = user_input['pregnancies'], orient = 'h', color = "purple")

plt.title('PREGNANCIES Distribution', fontsize=14)
plt.xlabel('Values')
#sns.stripplot.set_label('User Input')
#plt.legend(labels=["Boxplot","User Input"], loc = 2, bbox_to_anchor = (1,1))#, "bbox=props)#, color="purple")
#fig1, ax1 = ax1.get_legend_handles_labels()
#ax1.legend()
#sns.despine

ax1.set(xlim=(0,18))

props = dict(boxstyle='round', facecolor='plum', alpha=0.2)
ax1.text(0, 0.4, "Purple Dot = User Input", fontsize=8, bbox=props)
#ax1.text(4, 0.4, "            Couple of times            ", fontsize=10, bbox=props)
#ax1.text(8, 0.4, "                   Lots of times                   ", fontsize=10, bbox=props)
#ax1.text(13.1, 0.4, "               OMG times                ", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig1)

###   GLUCOSE   ###
fig2, ax2 = plt.subplots(figsize=(10,2))
sns.boxplot(df['Glucose'], color="red", width=.5)
sns.stripplot(data = user_input['glucose'], orient = 'h', color = "black")

plt.title('GLUCOSE Distribution', fontsize=14)
plt.xlabel('Values in mg/dL')

ax2.set(xlim=(0,200))

props = dict(boxstyle='round', facecolor='red', alpha=0.2)
ax2.text(72, 0.4, " Acceptable ", fontsize=10, bbox=props)
ax2.text(99, 0.4, " Good Concentration", fontsize=10, bbox=props)
ax2.text(140, 0.4, "OK if 2h post eating", fontsize=10, bbox=props)
ax2.text(170, 0.4, "Too much", fontsize=10, bbox=props)


plt.tight_layout()
st.pyplot(fig2)

###   BLOOD PRESSURE   ###
fig3, ax3 = plt.subplots(figsize=(10,2))
sns.boxplot(df['BloodPressure'], color="grey", width=.5)
sns.stripplot(data = user_input['blood_pressure'], orient = 'h', color = "red")

plt.title('BLOOD PRESSURE Distribution', fontsize=14)
plt.xlabel('Values in mmHg')

ax3.set(xlim=(0,250))

props = dict(boxstyle='round', facecolor='black', alpha=0.2)
ax3.text(80, 0.4, " Normal Tension ", fontsize=10, bbox=props)
ax3.text(120, 0.4, " Tension too high ", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig3)

###   SKIN THICKNESS   ###
fig4, ax4 = plt.subplots(figsize=(10,2))
sns.boxplot(df['SkinThickness'], color="plum", width=.5)
sns.stripplot(data = user_input['skin_thickness'], orient = 'h', color = "purple")

plt.title('SKIN THICKNESS Distribution', fontsize=14)
plt.xlabel('Values in mm')

ax4.set(xlim=(0,100))

props = dict(boxstyle='round', facecolor='plum', alpha=0.2)
ax4.text(2.5, 0.4, "   Normal for men   ", fontsize=10, bbox=props)
ax4.text(18, 0.4, "   Normal for women   ", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig4)

###   INSULIN   ###
fig5, ax5 = plt.subplots(figsize=(10,2))
sns.boxplot(df['Insulin'], color="red", width=.5)
sns.stripplot(data = user_input['insulin'], orient = 'h', color = "black")

plt.title('INSULIN Distribution', fontsize=14)
plt.xlabel('Values in mlU/L')

ax5.set(xlim=(0,999))

props = dict(boxstyle='round', facecolor='red', alpha=0.2)
ax5.text(20, 0.4, "Acceptable Insulin Level", fontsize=10, bbox=props)
ax5.text(160, 0.4, "High Insulin Level", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig5)

###   BMI   ###
fig6, ax6 = plt.subplots(figsize=(10,2))
sns.boxplot(df['BMI'], color="grey", width=.5)
sns.stripplot(data = user_input['BMI'], orient = 'h', color = "red")

plt.title('BODY MASS INDEX Distribution', fontsize=14)
plt.xlabel('Values')

ax6.set(xlim=(0,70))

props = dict(boxstyle='round', facecolor='black', alpha=0.2)
ax6.text(18, 0.4, "Thin", fontsize=10, bbox=props)
ax6.text(24.9, 0.4, "Normal", fontsize=10, bbox=props)
ax6.text(29.9, 0.4, "Thick", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig6)

###   DPF   ###
fig7, ax7 = plt.subplots(figsize=(10,2))
sns.boxplot(df['DiabetesPedigreeFunction'], color="plum", width=.5)
sns.stripplot(data = user_input['DPF'], orient = 'h', color = "purple")

plt.title('DIABETES PEDIGREE F° Distribution', fontsize=14)
plt.xlabel('Values in %')

ax7.set(xlim=(0,2.500))

props = dict(boxstyle='round', facecolor='plum', alpha=0.2)
ax7.text(.050, 0.4, "50% diabete risk", fontsize=10, bbox=props)
ax7.text(.100, 0.4, "100% diabete risk", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig7)

###   AGE   ###
fig8, ax8 = plt.subplots(figsize=(10,2))
sns.boxplot(df['Age'], color="red", width=.5)
sns.stripplot(data = user_input['age'], orient = 'h', color = "black")

plt.title('AGE Distribution', fontsize=14)
plt.xlabel('Values')

ax8.set(xlim=(0,110))

props = dict(boxstyle='round', facecolor='red', alpha=0.2)
#ax8.text(2, 0.4, "   Few times   ", fontsize=10, bbox=props)

plt.tight_layout()
st.pyplot(fig8)


###########      FIN - BoxPlots      ##########

# faire grid de scatter plots


# Function for Accessing Performance (Confusion Matrix, ROC Curve and Precision-Recall Curve)
#def plot_metrics1(metrics_list1):
#    if "Confusion Matrix" in metrics_list1:
#        st.subheader("Confusion Matrix")
 #       plot_confusion_matrix(gb_model, X_test, y_test)
 #       st.pyplot()
        
#def plot_metrics2(metrics)
#    if "ROC Curve" in metrics_list:
#        st.subheader("ROC Curve")
#        plot_roc_curve(gb_model, X_test, y_test)
#        st.pyplot()
#    if "Precision-Recall Curve" in metrics_list:
#        st.subheader("Precision-Recall Curve")
#        plot_precision_recall_curve(gb_model, X_test, y_test)
#        st.pyplot()
#class_names = ["edible", "poisnous"]

#Show the models metrics
st.subheader('Accessing Performance:')
gb_model = GradientBoostingClassifier
prediction = GradientBoostingClassifier.predict(user_input) #Store the models prediction in a variable
#accuracy = gb_model.score(X_test, y_test)
#y_pred = gb_model.predict(X_test)
#st.write("Results Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
#st.write("Model Accuracy: ", accuracy.round(3))
#st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))

def confusion_matrix_plot (y_test, prediction2):
    
    cm = confusion_matrix(y_test, prediction2)
    classes = ['0', '1']
    figure, ax = plot_confusion_matrix(conf_mat = cm,
                                       class_names = classes,
                                       show_absolute = True,
                                       show_normed = False,
                                       colorbar = True)

prediction2 = gb_model.predict(X_test)
fig = confusion_matrix_plot(y_test, prediction2)
st.pyplot(fig)

st.write("Model Accuracy: ", str(accuracy_score(y_test, GradientBoostingClassifier.predict(X_test)).round(3) * 100) + '%' )
#st.write("Recall Precision: ", str(recall_score(y_test, GradientBoostingClassifier.predict(X_test)).round(3) * 100) + '%' )
#metrics = ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
#st.subheader("Confusion Matrix")
#fig1 = plot_confusion_matrix(gb_model)
#st.pyplot(fig1)

#create a bar chart for features importance
st.subheader('Feature Importance')

#Create a function to plot feature importances
@st.cache(persist= True)
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    
fig9 = plot_feature_importance(gb_model.feature_importances_,X_train.columns,'GB Classifier ')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig9)

#set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)
st.write("Precision Score: ", str(precision_score(y_test, GradientBoostingClassifier.predict(X_test)).round(3) * 100) + '%' )


#plt.rcParams["figure.figsize"] = (20, 10)
#df.hist(grid=False, alpha=0.5)
#fig = df.hist(grid=False, alpha=0.5)
#st.pyplot(fig)