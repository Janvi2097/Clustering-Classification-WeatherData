# Clustering-Classification-WeatherData
 Application of Clustering (Gaussian Mixture and Mean-Shift), Classification (SVM and Naive Bayes) techniques for Weather Prediction

## Introduction
 I have used two datasets for clustering and classification. I have implemented Weather dataset in clustering which includes Air pressure, air temperature, humidity, rainfall, rain duration, wind speed etc. However, for the classification I have analysed the relation between humidity and other attributes. I have implemented the following clustering and classification techniques for this dataset:
i.	Gaussian Mixture Clustering
ii.	Mean-Shift Clustering
iii. Support Vector Machines Classification
iv.	Naïve Bayes Classification

## Result of Clutering:
### Gaussian Mean Clustering
![GM](https://user-images.githubusercontent.com/31371838/103689556-78ada500-4f61-11eb-9c28-bdec72bd2949.png)
### Mean Shift Clustering
![MS](https://user-images.githubusercontent.com/31371838/103689558-79463b80-4f61-11eb-886f-40e170942003.png)

## Result of Classification:
### SVM Confusion Matrix and F-Score
![SVM_CM](https://user-images.githubusercontent.com/31371838/103689725-b01c5180-4f61-11eb-8671-71a0fff192e2.png)
![SVM_FScore](https://user-images.githubusercontent.com/31371838/103689726-b01c5180-4f61-11eb-9f38-0bbd399a1dcf.png)
### Naive Bayes Confusion Matrix and F-Score
![NB_CM](https://user-images.githubusercontent.com/31371838/103689770-bca0aa00-4f61-11eb-97a9-9465910b1b3e.png)
![NB_FScore](https://user-images.githubusercontent.com/31371838/103689768-bc081380-4f61-11eb-87d8-53ed36d7cb01.png)

## Conclusion:
I would like to point out a few things for clustering:
i. There 8 clusters which give more insight about the distribution of air temperature over humidity whereas in the Gaussian Mixture the moderate air temperature is distributed between low and high humidity.
ii.	Although the tendency of mean shift is to draw data points with lesser density towards the cluster of higher density, the output obtained is more accurate without any signs of over-clustering as some outliers which are very close to each other are still distributed under different clusters


The conclusion of classification is as follows:
The Naïve Bayes method uses probabilistic methods to predict values whereas the SVM uses text categorization to perform the classification. It is not necessary that all the data classified using probabilistic methods is correct. Moreover, the humidity data here is categorical hence, the accuracy is higher in SVM as takes into consideration the association of categorical values with the numerical values to produce a better result. The Naïve Bayes predicts features using the probabilistic methods and considers other attributes independent which does not benefit our motive to get accurate results as the weather data is dependent on multiple attributes of the collected data.
