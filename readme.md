# Fire Automaton Implementation
This repository is an implementation of a cellular automata that is computed on a GPU using CUDA.

## What is the fire automaton problem in computer science?
The fire automata problem is a cellular automata that is used to simulate the spread of fire in a forest. 
The forest is represented by a grid of cells, where each cell can be either empty, a tree, or on fire. 
The automaton evolves over time according to a set of rules that determine how the fire spreads from one site to another.

## The Algorithm
* An cell with a burning tree will become empty:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sC(t)=Fire → sC(t+1)=Empty <br>
* A cell containing a tree will catch on fire, if at least one neighbor is on fire: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sC(t) = Tree → sC(t+1)=Fire if ∈ NC where s’(t)=Fire<br>
* A cell containing a tree without a neighbor on fire will catch fire with a probability p
or stay a tree with a probability (1-p):<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sC(t) = Tree → sC(t+1)=Fire with probability p if ∈/ NC where s’(t)=Fire<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sC(t) = Tree → sC(t+1)=Tree with probability 1-p if ∈/ NC where s’(t)=Fire<br>
* An empty cell will grow a new tree with a probability g or stay empty with a probability (1-g): <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sC(t) = Empty → sC(t+1)=Tree with probability g <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sC(t) = Empty → sC(t+1)=Empty with probability 1-g

##Parallelization
To improve the performance of the algorithm, the computation is parallelized using CUDA JIT.
CUDA JIT is a feature of the Numba Python package that allows you to compile and execute Python functions on a NVIDIA GPU using the CUDA programming model. It allows you to write Python code that can take advantage of the parallel processing power of a GPU without having to write CUDA C or C++ code.

## Performance
1 Iterations elapsed time:  <s>1.063 s</s><br>
10 Iterations elapsed time:  0.045 s<br>
100 Iterations elapsed time:  0.485 s<br>
1000 Iterations elapsed time:  4.669 s<br>
10000 Iterations elapsed time:  45.004 s<br>

## The result
![](Video_2023_01_09-1_edit_2.gif)
