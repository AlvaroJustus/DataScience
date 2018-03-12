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


## Data Import and Preview: 
```shell
$ python
```
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
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAskAAAF1CAYAAAAa1Xd+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBo%0AdHRwOi8vbWF0cGxvdGxpYi5vcmcvpW3flQAAHSlJREFUeJzt3XuwZVddJ/DvDzoQSIAGQtqYRBMl%0AQZEZeXRhkBqnQ8BKEA0zJQyMDJFJVVsjUgqOGK15UePMgOOI4IOaHmBsBAltFEgYBFLBOw4WQRNE%0AXkETnukkpgHzajAB5Dd/nN1yXWnSt5N77knf8/lUnTp7r7PO2b99V/Xp7113nX2quwMAAHzDfRZd%0AAAAA3NsIyQAAMBCSAQBgICQDAMBASAYAgIGQDAAAAyEZ4B6qqh+vqvet2t9fVd+xgcf/WFXt2Kjj%0AASwDIRlYClOQ/UhVfbmq/rqqfquqHjKPY3X3sd39qem4v11Vv3R3X6uqPlNVTx3a/kEo7+7v6e6V%0AQ7zOKVXVVbXl7tYCsEyEZGDTq6qfTfKKJD+X5CFJzkhySpL3VNVRCyxt0xC+gc1GSAY2tap6cJKX%0AJXlRd7+ru7/a3Z9J8uwkpyb5l1O/fzDjW1U7qmrvqv0LquqTVXVbVX28qv7ZXRyzq+qRVbUzyY8l%0Aeem0BOOSqvq5qvr9of+vV9Wv3YNz/PvZ5qp6YlVdUVW3VtWNVfWrU7c/nu5vnmp5UlXdp6r+XVV9%0Atqr2VdUbVs+uV9Xzp8e+WFX/fjjOf6qqi6rqjVV1a5Ifn479/qq6uapuqKrfqKr7DT+Xn6yqq6ef%0A43+uqu+cnnNrVe1Z3R9gkYRkYLP7/iRHJ/mD1Y3dvT/JHyb5wTW+zieT/JPMZqJfluSNVXXCXT2h%0Au3cleVOSX56WYPxwkjcmObuqtiZ/PwP7L5L8zprP6K69KsmruvvBSb4zyZ6p/Qem+61TLe9P8uPT%0A7cwk35Hk2CS/MdX16CS/lVnIPyGz8z5xONa5SS5KsnU6z79L8uIkxyV5UpKzkvzk8Jyzkzwhs9n8%0AlybZNR3j5CSPSfLce3DuAOtGSAY2u+OSfKG7v3aQx25I8oi1vEh3/153X9/dX+/utyS5OskTD7eY%0A7r4hs1ndZ01NZ0/1XXkXT3vbNDt7c1XdnFl4/Wa+muSRVXVcd+/v7svvou+PJfnV7v7U9EvDLyR5%0AzhTcfzTJJd39vu7+SpL/kKSH57+/u982/Uz+truv7O7Lu/tr02z9/0zyT4fnvKK7b+3ujyX5aJL3%0ATMe/JbNfWh53F/UCbBghGdjsvpDkuG+yZvaEJJ9fy4tMSw8+tCqoPiazAH537E7yvGn7eTn0LPIz%0Au3vrgVvuPDu72vlJTk/yiar6s6p6xl30/dYkn121/9kkW5Jsmx679sAD3f3lJF8cnn/t6p2qOr2q%0A3jF9MPLWJP81d/4Z3bhq+28Psn/sXdQLsGGEZGCze3+SO5L889WNVXVMknOS/N+p6UtJHriqy7es%0A6vvtSf5Xkp9K8vApqH40Sa3h+OPsa5K8Lck/rqrHJHlGZksV1kV3X93dz01yfGYfVrxoOteD1XF9%0Akm9ftf9tSb6WWXC9IclJBx6oqgckefh4uGH/NUk+keS0abnHL2ZtPyOAex0hGdjUpj/jvyzJr1fV%0A2VV1VFWdkuT3MptlPhBQP5Tk6VX1sKr6liQ/s+plDoTMzydJVb0gs5nktbgxs/W+q2u6PbO1vL+b%0A5E+7+3N349QOqqqeV1WP6O6vJ7l5av67zGr/+lDLm5O8uKpOrapjM5v5fcu0NOWiJD9cVd8/fZju%0AZTl04H1QkluT7K+q70ryb9brvAA2mpAMbHrd/cuZzWr+SpLbknw6s1njp3b3l6Zuv5PkL5J8Jsl7%0Akrxl1fM/nuR/ZDYrfWOSf5TkT9Z4+NclefS0TONtq9p3T6+zXh/YO+DsJB+rqv2ZfYjvOd19+7Rc%0A4r8k+ZOpljOSvH46/h9n9jO5PcmLkmRaM/yiJBdmNqt8W5J9mc3KfzP/NrOrhdyW2cz7W+6iL8C9%0AWnUf7C9wAJtXVf3rzGZGn7yes7iHWcO3ZbY04Vu6+9ZF1HA4ppnmmzNbSvHpRdcDMG8u/g4sne5+%0AfVV9NbPLw214SK6q+yR5SZIL780Buap+OMllmS2z+JUkH8lsph1g0xOSgaXU3eu9zGFNpg/R3ZjZ%0AlSTOXkQNh+HczJZjVJIrMlu64c+PwFKw3AIAAAY+uAcAAAMhGQAABveKNcnHHXdcn3LKKQs59pe+%0A9KUcc8wxCzk2G8MYLwfjvByM8+ZnjJfDIsf5yiuv/EJ3P+JQ/e4VIfmUU07JFVdcsZBjr6ysZMeO%0AHQs5NhvDGC8H47wcjPPmZ4yXwyLHuao+u5Z+llsAAMDgkCG5qh5VVR9adbu1qn5m+urWS6vq6un+%0A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%0A+p+f5KbufmSSV079AADgiHHIkNwz+6fdo6ZbJ3lKkoum9t1JnjltnzvtZ3r8rKqqdasYAADmrLr7%0A0J2q7pvkyiSPTPKbSf57ksun2eJU1clJ/rC7H1NVH01ydnfvnR77ZJLv6+4vDK+5M8nOJNm2bdsT%0ALrzwwvU7q8Owf//+HHvssQs5NhvDGC+Hm265NV+9z/0XXcaGOv5By3W+iXFeBt6zl8Mix/nMM8+8%0Asru3H6rflrW8WHf/XZLHVtXWJG9N8t0H6zbdH2zW+E5JvLt3JdmVJNu3b+8dO3aspZR1t7KykkUd%0Am41hjJfDnkveneuOPnXRZWyoZ+84fdElbDjjvPl5z14OR8I4H9bVLbr75iQrSc5IsrWqDoTsk5Jc%0AP23vTXJykkyPPyTJ36xHsQAAsBHWcnWLR0wzyKmqByR5apKrkvxRkh+dup2X5O3T9sXTfqbH39tr%0AWdMBAAD3EmtZbnFCkt3TuuT7JNnT3e+oqo8nubCqfinJnyd53dT/dUl+p6quyWwG+TlzqBsAAObm%0AkCG5uz+c5HEHaf9UkicepP32JM9al+oAAGABfOMeAAAMhGQAABgIyQAAMBCSAQBgICQDAMBASAYA%0AgIGQDAAAAyEZAAAGQjIAAAyEZAAAGAjJAAAwEJIBAGAgJAMAwEBIBgCAgZAMAAADIRkAAAZCMgAA%0ADIRkAAAYCMkAADAQkgEAYCAkAwDAQEgGAICBkAwAAAMhGQAABkIyAAAMhGQAABgIyQAAMBCSAQBg%0AICQDAMBASAYAgIGQDAAAAyEZAAAGQjIAAAyEZAAAGAjJAAAwEJIBAGBwyJBcVSdX1R9V1VVV9bGq%0A+ump/WFVdWlVXT3dP3Rqr6p6dVVdU1UfrqrHz/skAABgPa1lJvlrSX62u787yRlJXlhVj05yQZLL%0Auvu0JJdN+0lyTpLTptvOJK9Z96oBAGCODhmSu/uG7v7gtH1bkquSnJjk3CS7p267kzxz2j43yRt6%0A5vIkW6vqhHWvHAAA5uSw1iRX1SlJHpfkA0m2dfcNySxIJzl+6nZikmtXPW3v1AYAAEeELWvtWFXH%0AJvn9JD/T3bdW1TftepC2Psjr7cxsOUa2bduWlZWVtZayrvbv37+wY7MxjPFyOOrrd+TE2z+96DI2%0A1MrK9YsuYcMZ583vpltuzZ5L3r3oMjbU8Q+6/6JL2HBHwv/NawrJVXVUZgH5Td39B1PzjVV1Qnff%0AMC2n2De1701y8qqnn5TkTv/Cu3tXkl1Jsn379t6xY8fdO4N7aGVlJYs6NhvDGC+HPZe8O9cdfeqi%0Ay9hQz95x+qJL2HDGefMzxsvhSPi/eS1Xt6gkr0tyVXf/6qqHLk5y3rR9XpK3r2p//nSVizOS3HJg%0AWQYAABwJ1jKT/OQk/yrJR6rqQ1PbLyZ5eZI9VXV+ks8ledb02DuTPD3JNUm+nOQF61oxAADM2SFD%0Acne/LwdfZ5wkZx2kfyd54T2sCwAAFsY37gEAwEBIBgCAgZAMAAADIRkAAAZCMgAADIRkAAAYCMkA%0AADAQkgEAYCAkAwDAQEgGAICBkAwAAAMhGQAABkIyAAAMhGQAABgIyQAAMBCSAQBgICQDAMBASAYA%0AgIGQDAAAAyEZAAAGQjIAAAyEZAAAGAjJAAAwEJIBAGAgJAMAwEBIBgCAgZAMAAADIRkAAAZCMgAA%0ADIRkAAAYCMkAADAQkgEAYCAkAwDAQEgGAICBkAwAAAMhGQAABkIyAAAMDhmSq+r1VbWvqj66qu1h%0AVXVpVV093T90aq+qenVVXVNVH66qx8+zeAAAmIe1zCT/dpKzh7YLklzW3acluWzaT5Jzkpw23XYm%0Aec36lAkAABvnkCG5u/84yd8Mzecm2T1t707yzFXtb+iZy5NsraoT1qtYAADYCFvu5vO2dfcNSdLd%0AN1TV8VP7iUmuXdVv79R2w/gCVbUzs9nmbNu2LSsrK3ezlHtm//79Czs2G8MYL4ejvn5HTrz904su%0AY0OtrFy/6BI2nHHe/IzxcjgS/m++uyH5m6mDtPXBOnb3riS7kmT79u29Y8eOdS5lbVZWVrKoY7Mx%0AjPFy2HPJu3Pd0acuuowN9ewdpy+6hA1nnDc/Y7wcjoT/m+/u1S1uPLCMYrrfN7XvTXLyqn4nJVm+%0AX48AADii3d2QfHGS86bt85K8fVX786erXJyR5JYDyzIAAOBIccjlFlX15iQ7khxXVXuT/MckL0+y%0Ap6rOT/K5JM+aur8zydOTXJPky0leMIeaAQBgrg4Zkrv7ud/kobMO0reTvPCeFgUAAIvkG/cAAGAg%0AJAMAwEBIBgCAgZAMAAADIRkAAAZCMgAADIRkAAAYCMkAADAQkgEAYCAkAwDAQEgGAICBkAwAAAMh%0AGQAABkIyAAAMhGQAABgIyQAAMNiy6AJg3vbddkdeeelfLbqMDfXip52+6BIA4IhmJhkAAAZCMgAA%0ADIRkAAAYCMkAADAQkgEAYCAkAwDAQEgGAICBkAwAAAMhGQAABkIyAAAMhGQAABgIyQAAMBCSAQBg%0AICQDAMBASAYAgMGWRRcAALDMXnnpXy26hA33uKMWXcGhmUkGAICBkAwAAIOlX26x77Y7lurPHC9+%0A2umLLgEA4F5vLjPJVXV2Vf1lVV1TVRfM4xgAADAv6x6Sq+q+SX4zyTlJHp3kuVX16PU+DgAAzMs8%0AZpKfmOSa7v5Ud38lyYVJzp3DcQAAYC7mEZJPTHLtqv29UxsAABwR5vHBvTpIW9+pU9XOJDun3f1V%0A9ZdzqGUtjkvyhQUde8O9ZNEFLMZSjXFinJeFcV4OSzjOSzfGS2qR4/zta+k0j5C8N8nJq/ZPSnL9%0A2Km7dyXZNYfjH5aquqK7ty+6DubHGC8H47wcjPPmZ4yXw5EwzvNYbvFnSU6rqlOr6n5JnpPk4jkc%0ABwAA5mLdZ5K7+2tV9VNJ3p3kvkle390fW+/jAADAvMzly0S6+51J3jmP156DhS/5YO6M8XIwzsvB%0AOG9+xng53OvHubrv9Jk6AABYanP5xj0AADiSLWVIrqqjq+pPq+ovqupjVfWyRdfE/FTVfavqz6vq%0AHYuuhfmoqs9U1Ueq6kNVdcWi62H9VdXWqrqoqj5RVVdV1ZMWXRPrq6oeNf0bPnC7tap+ZtF1sf6q%0A6sVT/vpoVb25qo5edE0Hs5TLLaqqkhzT3fur6qgk70vy0919+YJLYw6q6iVJtid5cHc/Y9H1sP6q%0A6jNJtne3a6tuUlW1O8n/6+7XTldOemB337zoupiPqrpvkuuSfF93f3bR9bB+qurEzHLXo7v7b6tq%0AT5J3dvdvL7ayO1vKmeSe2T/tHjXdlu+3hSVQVScl+aEkr110LcDdU1UPTvIDSV6XJN39FQF50zsr%0AyScF5E1rS5IHVNWWJA/MQb5P495gKUNy8vd/gv9Qkn1JLu3uDyy6Jubi15K8NMnXF10Ic9VJ3lNV%0AV07f5snm8h1JPp/kf09Lp15bVccsuijm6jlJ3rzoIlh/3X1dkl9J8rkkNyS5pbvfs9iqDm5pQ3J3%0A/113PzazbwR8YlU9ZtE1sb6q6hlJ9nX3lYuuhbl7cnc/Psk5SV5YVT+w6IJYV1uSPD7Ja7r7cUm+%0AlOSCxZbEvEzLaX4kye8tuhbWX1U9NMm5SU5N8q1Jjqmq5y22qoNb2pB8wPQnu5UkZy+4FNbfk5P8%0AyLRe9cIkT6mqNy62JOahu6+f7vcleWuSJy62ItbZ3iR7V/3F76LMQjOb0zlJPtjdNy66EObiqUk+%0A3d2f7+6vJvmDJN+/4JoOailDclU9oqq2TtsPyGzAPrHYqlhv3f0L3X1Sd5+S2Z/u3tvd98rfVrn7%0AquqYqnrQge0kP5jko4utivXU3X+d5NqqetTUdFaSjy+wJObrubHUYjP7XJIzquqB04UUzkpy1YJr%0AOqi5fOPeEeCEJLunT8/eJ8me7nZ5MDgybUvy1tl7bbYk+d3uftdiS2IOXpTkTdOf4j+V5AULroc5%0AqKoHJnlakp9YdC3MR3d/oKouSvLBJF9L8ue5l3773lJeAg4AAO7KUi63AACAuyIkAwDAQEgGAICB%0AkAwAAAMhGQAABkIyAAAMhGQAABgIyQAAMPj/oQGrm6Gv2J8AAAAASUVORK5CYII=">

#### *Try your first TensorFlow program*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```

## Contribution guidelines

**If you want to contribute to TensorFlow, be sure to review the [contribution
guidelines](CONTRIBUTING.md). This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs. So please see
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss) for general questions
and discussion, and please direct specific questions to [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in open-source software development:

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)

## For more information

* [TensorFlow Website](https://www.tensorflow.org)
* [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)
* [TensorFlow Course at Stanford](https://web.stanford.edu/class/cs20si)

Learn more about the TensorFlow community at the [community page of tensorflow.org](https://www.tensorflow.org/community) for a few ways to participate.

## License

[MIT](LICENSE)
