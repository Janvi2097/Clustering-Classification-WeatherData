from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from numpy import unique
import seaborn as sb
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.preprocessing import OrdinalEncoder
from sklearn.mixture import GaussianMixture
#from matplotlib import pyplot

print("--------------PART 1: Importing Data and cleaning----------------")
#importing file using pandas
weather = pd.read_csv(r'Weather.csv')
df = pd.DataFrame(weather)
df =df.dropna()
df1 = df.drop(columns=['rowID', 'hpwren_timestamp'])
df1 = df1[:5000]

print("--------------PART 2: Clustering----------------")

ec = OrdinalEncoder()
df2 = df1[['air_temp','relative_humidity']].copy()
print(df2)
df_transformed = ec.fit_transform(df2)
X=df_transformed

# Define the Gaussian model

model = GaussianMixture(n_components=5)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1],label=cluster)
# show the plot
plt.title("Air temperature vs Humidity")
plt.legend()
plt.xlabel("Air Temperature")
plt.ylabel("Humidity")
plt.show()

#-------------------------Meanshift Clustering---------------------------------------------------
# Estimate bandwith
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=60)

# Fit Mean Shift with Scikit
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(X)
labels = meanshift.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#  Predict the cluster for all the samples
P = meanshift.predict(X)

# Generate scatter plot for training data
colors = list(map(lambda x: 'r' if x == 1 else 'g' if x == 2 else 'b' if x == 3 else 'c' if x == 4 else 'm' if x == 5 else 'y' if x == 6 else '#2f4f4f', P))
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.title(f'Estimated number of clusters = {n_clusters_}')
plt.legend()
plt.xlabel('Air temperature')
plt.ylabel('Humidity')
plt.show()

print("--------------PART 2: Classification----------------")

data = pd.read_csv(r'C:\Users\prera\Downloads\Data Science with Python\A3\Janvi\weather_data.csv')
data = data.loc[1:20000,['datetime_utc', ' _conds', ' _dewptm', ' _fog', ' _hail', ' _hum', ' _pressurem', ' _rain', ' _snow',' _tempm', ' _thunder', ' _tornado', ' _vism',' _wdire']]

#renaming columns to convert into understandable names
weather_df = data.rename(index=str, columns={' _conds': 'condition', ' _dewptm':'dew', ' _fog':'fog',' _hail':'hail', ' _hum': 'humidity', ' _pressurem': 'pressure',' _rain':'rain',' _snow':'snow',' _tempm': 'temperature', ' _thunder':'thunder', ' _tornado': 'tornado', ' _vism':'visibility',' _wdire':'wind_direction'})


#replacing NaN with previous valid values
weather_df.ffill(inplace=True)
weather_df.dropna()

#sampling the dataset to 2000 random samples
weather_df = weather_df.sample(n=2000)
weather_df = weather_df.drop('datetime_utc',axis=1)
print('The weather dataset used here contains',weather_df.shape[0],'rows and',weather_df.shape[1],'columns')

#Grouping
humidity_label = ['very low','low','moderate','high','very high']
weather_df['humidity_n'] = pd.qcut(weather_df['humidity'], q=5,labels = humidity_label)


lab_enc = preprocessing.LabelEncoder()
weather_df['humidity_n'] = lab_enc.fit_transform(weather_df['humidity_n'])
s_x = weather_df.drop('humidity_n',axis=1)._get_numeric_data()
print(s_x)
#imputing missing values with mean
weather_df.fillna(weather_df.mean(), inplace = True)
s_y = weather_df['humidity_n']
s_y.astype('int')
scaler = StandardScaler()
scaler.fit(s_x)
s_x = scaler.transform(s_x)
s_x_train, s_x_test, s_y_train, s_y_test = train_test_split(s_x, s_y, test_size=0.2, random_state=1)
svc=SVC(kernel='rbf') #Default hyperparameters
svc.fit(s_x_train, s_y_train)
s_y_pred=svc.predict(s_x_test)
print('Accuracy Score:')
print(accuracy_score(s_y_test,s_y_pred))
print(classification_report(s_y_test,s_y_pred))
print('The Confusion Matrix for SVM is: \n')
print(confusion_matrix(s_y_test,s_y_pred))
plot_confusion_matrix(svc,s_x_test,s_y_test,display_labels=humidity_label)
plt.title('Confusion Matrix of SVM')
plt.show()

# #Predicted Values to 

ax1=plt.hist(s_y_test,histtype='barstacked',color='r',label='actual value')
ax2=plt.hist(s_y_pred,histtype='barstacked',color='blue',label='predicted value')
plt.title('Predicted and Actual Values of SVM')
plt.show()

#-------------------------------------------------NAIVE BAYES-------------------------------
lab_enc = preprocessing.LabelEncoder()
weather_df['humidity_n'] = lab_enc.fit_transform(weather_df['humidity_n'])
weather_df['humidity_n'] = pd.qcut(weather_df['humidity'], q=5,labels = humidity_label)

y = weather_df['humidity_n']
X = weather_df.drop('humidity_n',axis=1)._get_numeric_data()


X.fillna(X.mean(), inplace = True)
y = weather_df['humidity_n']
#y.astype('int')
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

#Gaussian Naive Bayes classifier

nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
#print("confussion matrix")
#print(nb_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')
print(classification_report(y_test,nbpred))

#Plot Confusion Matrix
plot_confusion_matrix(nb,X,y,display_labels=humidity_label,cmap='Blues')
plt.title('Confusion Matrix of Naive Bayes')
plt.show()

The interpreter of the python tries to convert everything to the binary number
system. The reason why numbers like 0.5, 0.25, 0.125 etc give the accurate results
and most other numbers do not is the fact that these numbers are sometimes
estimated to use them under the required decimal places.

The numbers like 1/10 or 1/20 are exactly divisible as they can be converted
 to decimal format exactly. However, numbers like 1/3, 1/7 or 22/7 are not 
convertible to the exact decimal form.
For an example in decimal number system, 0.05 is a perfect
representation of 1/20, but to represent 1/3, we need to shorten it to 0.33
which is not completely accurate but an estimate to represent it as a decimal
number. 
Similarly, in binary, floating numbers must be 1/2, 1/4, 1/8, 1/16 and so on
to get an exact binary result, (0.1,0.01,0.001,0.0001 in order) others
cannot be represented accurately and need an unlimited string of digits. This can also be called as floating point arithmetic.