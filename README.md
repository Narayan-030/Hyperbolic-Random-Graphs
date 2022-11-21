# Hyperbolic-Random-Graphs
The following code is a naive way to generate hyperbolic random graphs. The time complexity is O(N^2).
Hence it is recommended that the number of nodes be 10^3(or lesser) while changing the variables and validating the correctness of the code.
For N =10^4 number of nodes the code takes longer time to run(>30 mins!).

The input of the code is:
N = Number of nodes
T = Temperature
Gamma =  exponent of the power-law degree distribution
K = average degree

The output of the code is:

A pdf containing the visualisation of the network.
An adjecency list.
The log-log degree distribution of the network.
The average degree of the network.
For example; for the following input parameter values we get the following RHG.
N = 1000, K = 10, T = 0.5, Gamma = 2.5
![output](https://user-images.githubusercontent.com/86014109/203096897-d3833a72-3b4a-48fb-996c-ed1b9efd3bd8.png)

References:

Aldecoa, Rodrigo, Chiara Orsini, and Dmitri Krioukov. "Hyperbolic graph generator." Computer Physics Communications 196 (2015): 492-496.
