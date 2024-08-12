from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


matsize = (4,4)


if rank == 0:
    if (matsize[0]*matsize[1]) % size != 0 :
        print("Please run with a number of processes that divides the matrix size. Or change the size of square matrix accordingly. (line 9)")
        print("Current matrix size is ", matsize," and number of processes is ", size)
        exit()

    if (matsize[0] != matsize[1]):
        print("Please run with a square matrix. (line 9)")
        exit()


sqrt_size = int(np.sqrt(size)) 
# gives the number of rows and columns in the grid

subarray_dim = (matsize[0]//sqrt_size, matsize[1]//sqrt_size)
result = np.zeros(subarray_dim, dtype=np.int32)

if rank == 0:
    A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    B = np.array([[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]])

    print("A : \n", A)
    print("B : \n", B)

    res = A @ B 
    print("\nEXpected :\n", res)
    
    a = np.ascontiguousarray(A[0:subarray_dim[0], 0:subarray_dim[1]])
    b = np.ascontiguousarray(B[0:subarray_dim[0], 0:subarray_dim[1]])
    

    subsizes = subarray_dim
    order = MPI.ORDER_C
    starts = [[i,j] for i in range(0, matsize[0], subarray_dim[0]) for j in range(0, matsize[1], subarray_dim[1])]


    for i in range(1,size):
        vect = MPI.INT.Create_subarray(matsize, subsizes, starts[i], order=order)
        vect.Commit()    
        comm.Send([(np.frombuffer(A.data, np.int32, offset=0)), 1, vect], dest=i)
        vect.Free()
        vect = MPI.INT.Create_subarray(matsize, subsizes, starts[i], order=order)
        vect.Commit()    
        comm.Send([(np.frombuffer(B.data, np.int32, offset=0)), 1, vect], dest=i)
        vect.Free()

else :
    a = np.zeros(subarray_dim, dtype=np.int32)
    b = np.zeros(subarray_dim, dtype=np.int32)
    comm.Recv(a, source=0)
    comm.Recv(b, source=0)

comm.barrier()


dims = (sqrt_size, sqrt_size)
periods = (True, True)
reorder = False
crt2d = comm.Create_cart(dims, periods, reorder)
local_row, local_col = crt2d.Get_coords(rank)
left, right = crt2d.Shift(1, 1)
up, down = crt2d.Shift(0, 1)    

#making A1 and B1
for i in range(sqrt_size):
    if local_col == i:
        for j in range(i):
            crt2d.Sendrecv_replace(b, dest=up, source=down)

    if local_row == i:
        for j in range(i):
            crt2d.Sendrecv_replace(a, dest=left, source=right)


result = np.zeros(subarray_dim, dtype=np.int32)
result += np.dot(a, b)

for i in range(sqrt_size-1):
    crt2d.Sendrecv_replace(b, dest=up, source=down)
    crt2d.Sendrecv_replace(a, dest=left, source=right)
    result += np.dot(a, b)

comm.barrier()

if rank != 0:
    comm.Send(result, dest=0)

else:
    final_result = np.zeros(matsize, dtype=np.int32)  
    final_result[starts[0][0]:starts[0][0]+subarray_dim[0], starts[0][1]:starts[0][1]+subarray_dim[1]] = result
    
    for i in range(1,size):
        comm.Recv(result, source=i)
        final_result[starts[i][0]:starts[i][0]+subarray_dim[0], starts[i][1]:starts[i][1]+subarray_dim[1]] = result
    
    print("\nFinal result: \n", final_result)
    