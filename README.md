# Clustering-Classification-WeatherData
 Application of Clustering (Gaussian Mixture and Mean-Shift), Classification (SVM and Naive Bayes) techniques for Weather Prediction

## Introduction
 I have used two datasets for clustering and classification. I have implemented Weather dataset in clustering which includes Air pressure, air temperature, humidity, rainfall, rain duration, wind speed etc. However, for the classification I have analysed the relation between humidity and other attributes. I have implemented the following clustering and classification techniques for this dataset:
i.	Gaussian Mixture Clustering
ii.	Mean-Shift Clustering
iii. Support Vector Machines Classification
iv.	Naïve Bayes Classification

## Result of Clutering:


## Result of Classification:


## Conclusion:
I would like to point out a few things for clustering:
i. There 8 clusters which give more insight about the distribution of air temperature over humidity whereas in the Gaussian Mixture the moderate air temperature is distributed between low and high humidity.
ii.	Although the tendency of mean shift is to draw data points with lesser density towards the cluster of higher density, the output obtained is more accurate without any signs of over-clustering as some outliers which are very close to each other are still distributed under different clusters


The conclusion of classification is as follows:
The Naïve Bayes method uses probabilistic methods to predict values whereas the SVM uses text categorization to perform the classification. It is not necessary that all the data classified using probabilistic methods is correct. Moreover, the humidity data here is categorical hence, the accuracy is higher in SVM as takes into consideration the association of categorical values with the numerical values to produce a better result. The Naïve Bayes predicts features using the probabilistic methods and considers other attributes independent which does not benefit our motive to get accurate results as the weather data is dependent on multiple attributes of the collected data.