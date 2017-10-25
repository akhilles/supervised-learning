# Supervised Learning

CS 440 - Assignment 2  
Group: Akhil Velagapudi, Nithin Tammishetti

## Question 1

\newpage
## Question 2

### a)
The tree correctly categorizes all the provided examples.

### b)
Code is provided in `2-acceptance-tree/acceptance.py`.  
Algorithm generates the following tree:
```
GPA Class
|-- N (3.2 >= GPA)
|-- P (GPA >= 3.9)
|-- Published (3.9 > GPA > 3.2)
    |-- P (yes)
    |-- University (no)
        |-- N (rank 1)
        |-- N (rank 3)
        |-- P (rank 2)
```
using the following calculations to determine information gain:
```
Information gain:

['University', 'Published', 'Recommendation', 'GPA Class']
[0.11036014405977645, 0.006900300371591395, 0.11036014405977645, 0.6222849157562068]
Best attribute: GPA Class

['University', 'Published', 'Recommendation']
[0.17095059445466854, 0.4199730940219749, 0.0]
Best attribute: Published

['University', 'Recommendation']
[0.9182958340544896, 0.0]
Best attribute: University
```

### c)
The tree generated in part b is equivalent to the tree provided but this is a coincidence. It is possible for the algorithm to arrive at a decision tree that is simpler than the actual decision tree used to classify the samples. Also, there might be noise in the available data which the algorithm fails to ignore. 

\newpage
## Question 3
Code is provided in `3-svm/svm.py`.

### a)
![SVM Classification](3-svm/plot.png)

### b)
w = [1 1]<sup>T</sup>  
b = -4

\newpage
### c)
(no change)  
w = [1 1]<sup>T</sup>  
b = -4

![SVM Classification with Additional Data](3-svm/plot_new.png)

\newpage
## Question 4
Code is provided in `4-perceptron-learning/perceptron-learning.py`.

### a)
The alpha channel of each line represents the iteration. The darker the line, the later the iteration.
![Perceptron Learning](4-perceptron-learning/perceptron_2d.png)

### b)
Perceptron did reach a perfect classification.

\newpage
### c)
The alpha channel of each line represents the iteration. The darker the line, the later the iteration.

w = [0.200 -0.306]<sup>T</sup>  
error (proportion of misclassified samples) = 25%

Inputs are seperated at i1 = 0.6536

![Perceptron Learning (1D)](4-perceptron-learning/perceptron_1d.png)

\newpage
## Question 5

### a)
![Single Perceptron with Error](5-difficult-classification/5a.jpg)

\newpage
### b)

![Three Perceptrons without Error](5-difficult-classification/5bi.jpg)

![Multi-layer Perceptron](5-difficult-classification/5bii.jpg)