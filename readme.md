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

## The Implementation
```
@cuda.jit
def ArrayWorker2(grid, prob_array, p_grow, p_fire, next_grid):
    val_fire = 0xFF0000
    val_tree = 0x00FF00
    val_dirt = 0x0

    fire = 0
    num_rows, num_cols = grid.shape

    row, col = cuda.grid(2)
    current_val = grid[row, col]
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue

            # Compute the row and column indices of the current neighbor
            nb_row = row + i
            nb_col = col + j

            # Check if the neighbor is outside the grid
            if nb_row < 0 or nb_row >= num_rows or nb_col < 0 or nb_col >= num_cols:
                continue

            # Increment the number of fire if the neighbor is a fire
            fire += grid[nb_row, nb_col] == val_fire

    if current_val == val_dirt:
        if prob_array[row, col] < p_grow:
            next_grid[row, col] = val_tree
        else:
            next_grid[row, col] = val_dirt
    elif current_val == val_tree:
        if fire>0 or prob_array[row, col] < p_fire:
            next_grid[row, col] = val_fire
        else:
            next_grid[row, col] = val_tree
    elif current_val == val_fire:
        next_grid[row, col] = val_dirt


# Use torch.cuda to create the prob array
def create_prob_array(shape):
    shape = torch.Size(shape)
    x = torch.cuda.FloatTensor(shape)
    torch.rand(shape, out=x)
    return x

#Run the function on the GPU
def cuda_worker(array, p_grow, p_fire):
    array = cuda.to_device(array)
    array = cuda.as_cuda_array(array)
    prob_array = create_prob_array(array.shape)
    prob_array = cuda.as_cuda_array(prob_array)
    C_global_mem = cuda.device_array(array.shape, dtype=np.uint32)
    threadsperblock = (16, 16)
    blockspergrid_x = int(C_global_mem.shape[0] // threadsperblock[0])
    blockspergrid_y = int(C_global_mem.shape[1] // threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    ArrayWorker2[blockspergrid, threadsperblock](array, prob_array, p_grow, p_fire, C_global_mem)


    return C_global_mem.copy_to_host()
```
## The result
![](Video_2023_01_09-1_edit_2.gif)
