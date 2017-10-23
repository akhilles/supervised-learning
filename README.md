# Supervised Learning

CS 440 - Assignment 2  
Group: Akhil Velagapudi, Nithin Tammishetti

## Question 1


## Question 2

### a)
The tree correctly categorizes all the provided examples.

### b)
Code is provided in `2-acceptance-tree/acceptance.py`. Algorithm generates the following tree:
```
GPA Class
├── N (3.2 >= GPA)
├── P (GPA >= 3.9)
└── Published (3.9 > GPA > 3.2)
    ├── P (yes)
    └── University (no)
        ├── N (rank 1)
        ├── N (rank 3)
        └── P (rank 2)
```