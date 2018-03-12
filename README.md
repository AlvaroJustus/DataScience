<div align="center">
  <img src="http://cdn.wittysparks.com/wp-content/uploads/2017/08/09210754/Machine-Learning.jpg"><br><br>
</div>

-----------------


| **`Scikit-Learn`** |
|--------------------|
[![Scikit-Learn](https://elitedatascience.com/wp-content/uploads/2016/11/sklearn-logo.png)](http://scikit-learn.org)

# **Wine Quality Analysis**

## **Abstract**
This work aims to perform an exploratory analysis in a database on the quality of red wines. The database consists 
of wine samples from the north of Portugal, provided by the Viticulture Commission of the Vinho Verde Region (CVRVV). 
The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009](http://www3.dsi.uminho.pt/pcortez/wine/)) using machine learning techniques and models to classify and predict dataset data. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

## Source
* [Paulo Cortez, University of Minho, Guimarães, Portugal](http://www3.dsi.uminho.pt/pcortez).
* [A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @2009](http://www.vinhoverde.pt/en/).
* [UCI Machine Learning Repository Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).


## Data Information ##

|**Number of Instances:**| 4898      | **Attribute Characteristics:** |  Real    |
|------------------------|-----------|--------------------------------|----------|
|    **Date Donated:**   | 2009-10-07|          **Area:**             | Business |

* There are 12 classes that make up the dataset. There is no previous information about any class, so it is necessary to analyze the influence and relevance of each one.

**Attribute Information:**

For more information, read [Cortez et al., 2009](http://www3.dsi.uminho.pt/pcortez/wine/). 

Input variables (based on physicochemical tests):

* 1 - Fixed acidity (g(tartaric acid)/dm3)
* 2 - Volatile acidity (g(acetic acid)/dm3)
* 3 - Citric acid (g/dm3)
* 4 - Residual sugar (g/dm3)
* 5 - Chlorides (g(sodium chloride)/dm3)
* 6 - Free sulfur dioxide (mg/dm3)
* 7 - Total sulfur dioxide (mg/dm3)
* 8 - Density (g/cm3)
* 9 - pH
* 10 - Sulphates (g(potassium sulphate)/dm3)
* 11 - Alcohol (% vol.)

Output variable (based on sensory data):
* 12 - quality (score between 0 and 10)


[References](http://www3.dsi.uminho.pt/pcortez/wine5.pdf)

## Goal:

* Consists of identifying the model that best applies to classify and predict data using input classes to predict output (quality), using machine learning models in Python. Libraries will be used to provide support for the implementation of these models, such as Pandas, Seaborn and the famous Scikit-learn. For this, a preliminary and an exploratory analysis will be performed on the data, to understand the behavior of the variables, how their relationships are and what the best technique of machine learning to apply.

### Data Import and Preview: 

```python
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> %matplotlib inline
>>> import seaborn as sns
>>> wine = pd.read_csv('winequality-red.csv',sep=';')
>>> wine.describe()

	fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
count	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000	1599.000000
mean	8.319637	0.527821	0.270976	2.538806	0.087467	15.874922	46.467792	0.996747	3.311113	0.658149	10.422983	5.636023
std	1.741096	0.179060	0.194801	1.409928	0.047065	10.460157	32.895324	0.001887	0.154386	0.169507	1.065668	0.807569
min	4.600000	0.120000	0.000000	0.900000	0.012000	1.000000	6.000000	0.990070	2.740000	0.330000	8.400000	3.000000
25%	7.100000	0.390000	0.090000	1.900000	0.070000	7.000000	22.000000	0.995600	3.210000	0.550000	9.500000	5.000000
50%	7.900000	0.520000	0.260000	2.200000	0.079000	14.000000	38.000000	0.996750	3.310000	0.620000	10.200000	6.000000
75%	9.200000	0.640000	0.420000	2.600000	0.090000	21.000000	62.000000	0.997835	3.400000	0.730000	11.100000	6.000000
max	15.900000	1.580000	1.000000	15.500000	0.611000	72.000000	289.000000	1.003690	4.010000	2.000000	14.900000	8.000000

>>> wine.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
fixed acidity           1599 non-null float64
volatile acidity        1599 non-null float64
citric acid             1599 non-null float64
residual sugar          1599 non-null float64
chlorides               1599 non-null float64
free sulfur dioxide     1599 non-null float64
total sulfur dioxide    1599 non-null float64
density                 1599 non-null float64
pH                      1599 non-null float64
sulphates               1599 non-null float64
alcohol                 1599 non-null float64
quality                 1599 non-null int64
dtypes: float64(11), int64(1)
memory usage: 150.0 KB

>>> wine.head()

fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
0	7.4	0.70	0.00	1.9	0.076	11.0	34.0	0.9978	3.51	0.56	9.4	5
1	7.8	0.88	0.00	2.6	0.098	25.0	67.0	0.9968	3.20	0.68	9.8	5
2	7.8	0.76	0.04	2.3	0.092	15.0	54.0	0.9970	3.26	0.65	9.8	5
3	11.2	0.28	0.56	1.9	0.075	17.0	60.0	0.9980	3.16	0.58	9.8	6
4	7.4	0.70	0.00	1.9	0.076	11.0	34.0	0.9978	3.51	0.56	9.4	5

>>> sess.close()
```

## Data Exploration

```python
>>> wine['quality'].hist(figsize=(12,6), alpha=0.5)
    plt.title('Quality Histogram')
```
   ![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/1.png)
 
* As we can see, the quality of red wines in the dataset ranges from 3 to 8, with the highest number of notes being concentrated in 5 and 6. This means that the dataset contains more data about normal wines than good or bad wines.

```python
>>> plt.figure(figsize=(16,12))
sns.countplot(y='residual sugar',data=wine[wine['quality']>=7], palette='RdBu',hue='quality')
plt.title('Residual Sugar')
```
  ![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/2.png)
  
* By using the barplot by relating the amount of residual sugar by splitting the data between the wines between notes 7 and 8, it is possible to verify that the dataset concentrates data with a residual amount of sugar between 1.7 (g / dm3) and 2.5 (g / dm3) both for wines of note 7 and 8.

```python
>>> sns.jointplot(x='quality', y='residual sugar',data=wine,size=9, color='r',alpha=0.6)
```
![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/3.png)

* The graph presented above consists of the relation of the residual sugar and the quality attributed to the wine. In this graph it is possible to analyze where the data are concentrated when relating these two parameters and how their distributions are related. It is also possible to see the way the data divides, this already us clues as to the best method to apply in this dataset.

```python
>>> sns.barplot(x='quality',y='pH', data=wine)
```
![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/4.png)

* The figure above shows the relationship between the quality attributed to wines and their pH's. At first the pH does not seem to vary much and nor is it a very important factor to classify the quality of the wine, since the values do not vary much. But it is important to check this information, we will do this in sequence.

```python
>>> plt.figure(figsize=(12,6))
    sns.distplot(wine['pH'])
```
![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/5.png)

* As it is possible to see in the graph presented above, the pH in the dataset behaves almost like a normal dissolution, having a mean of 3.311113195747343, a minimum of 2.740000 and a maximum of 4.010000 with a standard deviation of 0.154386. The pH does not really seem to vary significantly in the samples, but even so, we will keep it in the implementation of the machine learning models.

```python
>>> sns.jointplot(x='quality',y='fixed acidity',data=wine,kind='hex', size=9)
```
![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/6.png)

* The above joint plot shows how the ratio of fixed acidity distributions (g (tartaric acid) / dm3) and wine quality is. It is possible to analyze that the concentration of fixed acidity is concentrated more between the values of 6 and 9 (g (tartaric acid) / dm3), with an average of 8.319637 (g (tartaric acid) / dm3). And that perhaps wines of notes 5 and 6 have a higher fixed acidity than the others (this may be due to the higher concentration of samples with these notes in the data).

```python
>>> plt.figure(figsize=(12,8))
    sns.heatmap(wine.corr(),cmap='coolwarm',lw=1)
```
![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/7.png)

* Finally a graph showing the correlation of the data in the dataset, in this graph we can verify if a linear regression model would apply well here (which was not the case to date). In this heatmap it is possible to verify the influence of one variable on the other and how strong or weak it is, I particularly like to make a heatmap for this analysis. It is possible to observe a strong positive correlation between critric acid and fixed acidity and between fixed acidity and density. A positive mean correlation between alcohol and quality. However, the most important analysis to take from this graph is that it is evident that a linear regression model would not have good results if applied in these data.

```python
>>> sns.pairplot(wine, hue='quality')
```
![My image](https://github.com/AlvaroJustus/Machine-Learning-Wine-Quality/blob/master/Docs/Images/8.png)

* Click on the image to enlarge it. Finally a pairplot, relating the distributions of all variances with each other, divided by the quality of the wines. I really like this chart and I consider it of utmost importance to check which method best applies to handle a dataset. You can see that there is a lot of data overlapping and a strong grouping, which means that it will be difficult to classify data so precisely. However, since quality is given discreetly and the output of the model is based on it, it may be possible to obtain good results without the need to transform the quantitative data (from 0 to 10) to qualitative (bad, intermediate and good) data.

## Contribution guidelines



## For more information



## License

[MIT](LICENSE)
