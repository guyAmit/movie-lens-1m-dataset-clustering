
Jupyter Notebook
readme for mini project data analysis Last Checkpoint: an hour ago (unsaved changes) Current Kernel Logo

Python 3

    File
    Edit
    View
    Insert
    Cell
    Kernel
    Widgets
    Help

# Clustering of the M1-movielens dataset

​

## clarifications:

this is a university assignment. the definitions below belongs to dr.sabto at the Ben gurion University. the implementation and the algorithms are mine ([guyAmit](https://github.com/guyAmit)) and yuval hadad(no github yet)

​

## goal:

>Cluster subsets of movies, from the data set according to the following objective function. i.e. find a partition of the subset $C = \{c_1,c_2,\ldots,c_m\}$ 

$ \forall c_i, c_j \in C: c_i \ c_j = \emptyset$ and $ Cost(C) \rightarrow min $.

​

### definitions: 

  - $p(m)$ - the probability that a user will rate the movie m

  - $p(m_i,m_j)$ - the probability  that a user will rate both the movies $m_i$ and $m_j$

  - objective function: a function that measure  the 'price' of a partition

  - $Cost(C) = \sum_{i=1}^n cost(c_i)$

  - $$ cost(c_i) =  \begin{cases}

  \sum_{m_i\in C} \sum_{m_j \neq m_i , m_j\in C} \frac {1}{|C| -1} \cdot log(\frac {1} {log(p(m_i,m_j))} &|C|\geq 2\\

  log(\frac {1} {p(m)}) &|C|=1 \text{ and } C=\{m\}\end{cases} $$

  

### probabilty estimations from the dataset:

- let $N$ be the number of users in the dataset

- let $k$ be the number of movies in the dataset

- let $n_i$ be the number of movies that user $i$ watched

- $v_i(j)=1$ if and only if user $i$ watched movie $m_j$ 

- $n_i = \sum_{j=1}^k v_i(j)$

- finly we define the probabilties to be: 

$$p(m_j) = \frac{1}{N+1}\cdot \big( \frac{2}{k} +\sum_{i=1}^N \frac {2}{n_i} \cdot v_i(j) \big) $$

$$p(m_j,m_t) = \frac{1}{N+1} \cdot \big(\frac{2}{k(k-1)} +\sum_{i=1}^N \frac {2}{n_i(n_i -1} \cdot v_i(j)v_i(t) \big )$$

​

### this repo include:

- a roport on the project

- several clustering algorithems

- in the final folder there is the final algorithm. to use it - download the folder and follow the insteractions below: > <br/>

 run $getmoviefiles$ <br/>

 run $moviecluster <someimput> <clustering method 1 or 2> <subset-path>$ <br/>

 1 for pivot algorithm <br/>

 2 for linear program based algorithm <br/>

​


