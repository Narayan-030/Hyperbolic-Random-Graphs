import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import integrate
import itertools

#INPUT THE PARAMETERS HERE!#########
#N = 10000                      #Number of nodes
#K = 10                       # Average degree
#T = 0.5                      #Temperature
#gamma = 2.5                  #Power law exponent
#####################################
N = int(input('Please input the number of nodes:',))
K = float(input('Please input the Average degree of the network:',))
T = float(input('Please input the Temperature:',))
gamma = float(input('Please input the power law exponent value:',))

curve = 1
if gamma<2 or gamma == np.inf:
    raise Exception("gamma belongs to the range [2,inf);Please input correct value of the parameter")
if T==0 or T== np.inf:
    raise Exception("T belongs to the range (0,inf);Please input correct value of the parameter")

if T<= 1:
    alpha = (gamma - 1)*curve/2
else:
    alpha = ((gamma - 1)*curve)/(2*T)


def r_func(R):
    func = lambda r,r1,t: alpha*np.exp(alpha*(r1-R))*alpha*np.exp(alpha*(r-R))*(1/(1+np.exp(curve*((np.arccosh(np.cosh(curve*r)*np.cosh(curve*r1)-np.sinh(curve*r)*np.sinh(curve*r1)*np.cos(np.pi-np.abs(np.pi-np.abs(t)))))/curve)-R)/(2*T)))
    y = integrate.nquad(func, [[0,R], [0,R], [0,np.pi]], full_output=True)
    return(K-(N/np.pi)*y[0])

def bisection(fn,a,b,e):
    if np.sign(fn(a))==np.sign(fn(b)):
        raise Exception("The range of R given do not bound a root")
    for i in range(200):
        m = (a + b)/2
        f1 = fn(m)
        if np.abs(f1)< e:
            return m
        elif np.sign(fn(a)) == np.sign(f1):
            a = m
        else:
            b = m

a = 1
b = 50
R = bisection(r_func,a,b,0.001)
Theta_i = np.random.uniform(0,2*np.pi,N).tolist()
U_i = np.random.uniform(0,1,N)
r_i = [(1/alpha)*np.arccosh(1+(np.cosh(alpha*R)-1)*i) for i in U_i]    

def connection_probability(r1,r2,t1,t2):
    p = (1/(1+np.exp(curve*((np.arccosh(np.cosh(curve*r1)*np.cosh(curve*r2)-np.sinh(curve*r1)*np.sinh(curve*r2)*np.cos(np.pi-np.abs(np.pi-np.abs(t1-t2)))))/curve)-R)/(2*T)))
    return(p)
Nodes = [i for i in range(N)]
Edges = list(itertools.combinations(Nodes,2))
G = nx.Graph()
G.add_nodes_from(Nodes)
for j in range(len(Edges)):
    x,y = Edges[j]
    if connection_probability(r_i[x],r_i[y],Theta_i[x],Theta_i[y])>= np.random.uniform(0,1):
        G.add_edge(x,y)

plt.figure(figsize=(20, 20))
pos = dict(zip(Nodes,[np.array([r_i[i]*np.cos(Theta_i[i]),r_i[i]*np.sin(Theta_i[i])])for i in Nodes]))
nx.draw_networkx_nodes(G, pos, nodelist=Nodes,node_size=15,node_color='blue',alpha=1,edgecolors='black')
nx.draw_networkx_edges(G, pos,edgelist=G.edges(),alpha=0.15,width=0.6)
plt.savefig('Random_Hyperbolic_Graph.pdf',facecolor='w',format='pdf',dpi=200,bbox_inches='tight')
plt.show()

nx.write_adjlist(G, "Adjecency_list.adjlist")

y = list(nx.degree_histogram(G))
x = [i for i in range(len(y))]
plt.figure(figsize=(8, 6)) 
plt.loglog(x,y,'o')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree distribution')
plt.savefig('degree distribution RHG.pdf',facecolor='w',format='pdf',dpi=200,bbox_inches='tight')
plt.show()

print('The average degree of the obtained RHG is:',np.average([[i for (j,i) in G.degree()]]))

