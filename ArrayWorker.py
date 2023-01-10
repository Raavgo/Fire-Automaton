import time

import numba
import numpy as np
from numba import cuda, jit, vectorize,uint32
from copy import deepcopy

import torch


@cuda.jit
def ArrayWorker(array, prob_array, p_grow, p_fire, out_array):
    val_fire = 0xFF0000
    val_tree = 0x00FF00
    val_dirt = 0x0
    #array = cuda.shared.array(array, dtype=uint32)

    width, height = array.shape[0], array.shape[1]

    y, x = cuda.grid(2)
    if x < width and y < height:
        if array[x, y] == val_dirt:
            if prob_array[x, y] < p_grow:
                out_array[x, y] = val_tree
            else:
                out_array[x, y] = val_dirt
        elif array[x, y] == val_tree:
            out = 0
            up = y-1 if y-1 >= 0 else y
            down = y+1 if y+1 < height else y
            left = x-1 if x > 0 else x
            right = x + 1 if x + 1 < width else x

            out += array[y][right] == val_fire
            out += array[up][right] == val_fire
            out += array[up][x] == val_fire
            out += array[up][left] == val_fire
            out += array[y][left] == val_fire
            out += array[down][left] == val_fire
            out += array[down][x] == val_fire
            out += array[down][right] == val_fire

            if prob_array[x, y] < p_fire or out > 0:
                out_array[x, y] = val_fire
            else:
                out_array[x, y] = val_tree
        elif array[x, y] == val_fire:
            out_array[x, y] = val_dirt


@cuda.jit
def ArrayWorker2(grid, prob_array, p_grow, p_fire, next_grid):
    val_fire = 0xFF0000
    val_tree = 0x00FF00
    val_dirt = 0x0

    neighbors = 0
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

            # Increment the number of live neighbors if the neighbor is alive
            neighbors += grid[nb_row, nb_col] == val_fire

    if current_val == val_dirt:
        if prob_array[row, col] < p_grow:
            next_grid[row, col] = val_tree
        else:
            next_grid[row, col] = val_dirt
    elif current_val == val_tree:
        if neighbors>0 or prob_array[row, col] < p_fire:
            next_grid[row, col] = val_fire
        else:
            next_grid[row, col] = val_tree
    elif current_val == val_fire:
        next_grid[row, col] = val_dirt


def create_prob_array(shape):
    shape = torch.Size(shape)
    x = torch.cuda.FloatTensor(shape)
    torch.rand(shape, out=x)
    return x


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

import timeit

if __name__ == "__main__":

    setup = """import numpy as np;from ArrayWorker import cuda_worker;a = np.zeros((1024, 1024), dtype=np.uint32);p_grow = 0.001;p_fire = 0.0001"""
    start = time.time()
    import numpy as np
    from ArrayWorker import cuda_worker
    a = np.zeros((1024, 1024), dtype=np.uint32)
    p_grow = 0.001
    p_fire = 0.0001
    cuda_worker(a, p_grow, p_fire)
    print(f'1 Iterations', "elapsed time: ", round(time.time() - start, 3), "s")


    for e in [10, 100, 1000, 10000]:
        res = timeit.timeit(stmt="cuda_worker(a, p_grow, p_fire)",setup=setup, number=int(e))
        #print round res to 3 decimal places
        print(f'{e} Iterations', "elapsed time: ", round(res, 3), "s")



