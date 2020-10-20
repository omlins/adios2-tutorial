import numpy as np
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import adios2

def update_halo(A, neighbors_x, neighbors_y):
    if neighbors_x[0] >= 0: # MPI_PROC_NULL?
        sendbuf = np.copy(A[1,:]).flatten()
        recvbuf = np.zeros(A.shape[1])
        comm.Send(sendbuf, neighbors_x[0], 0)
        comm.Recv(recvbuf,  neighbors_x[0], 1)
        A[0,:] = recvbuf
    if neighbors_x[1] >= 0: # MPI_PROC_NULL?
        sendbuf = np.copy(A[-2,:]).flatten()
        recvbuf = np.zeros(A.shape[1])
        comm.Send(sendbuf, neighbors_x[1], 1)
        comm.Recv(recvbuf, neighbors_x[1], 0)
        A[-1,:] = recvbuf
    if neighbors_y[0] >= 0: # MPI_PROC_NULL?
        sendbuf = np.copy(A[:,1]).flatten()
        recvbuf = np.zeros(A.shape[0])
        comm.Send(sendbuf, neighbors_y[0], 2)
        comm.Recv(recvbuf,  neighbors_y[0], 3)
        A[:,0] = recvbuf
    if neighbors_y[1] >= 0: # MPI_PROC_NULL?
        sendbuf = np.copy(A[:,-2]).flatten()
        recvbuf = np.zeros(A.shape[0])
        comm.Send(sendbuf, neighbors_y[1], 3)
        comm.Recv(recvbuf, neighbors_y[1], 2)
        A[:,-1] = recvbuf

# MPI
nprocs      = MPI.COMM_WORLD.Get_size()
dims        = MPI.Compute_dims(nprocs, [0,0])
comm        = MPI.COMM_WORLD.Create_cart(dims)
me          = comm.Get_rank()
coords      = comm.Get_coords(me)
neighbors_x = comm.Shift(direction = 0,disp=1)
neighbors_y = comm.Shift(direction = 1,disp=1)

# Physics
lam        = 1.0                 # Thermal conductivity
cp_min     = 1.0                 # Minimal heat capacity
lx, ly     = 10.0, 10.0          # Length of computational domain in dimension x and y

# Numerics
nx, ny     = 128, 128            # Number of gridpoints in dimensions x and y
nt         = 10000               # Number of time steps
nx_g       = dims[0]*(nx-2) + 2  # Number of gridpoints of the global problem in dimension x
ny_g       = dims[1]*(ny-2) + 2  # ...                                        in dimension y
dx         = lx/(nx_g-1)         # Space step in dimension x
dy         = ly/(ny_g-1)         # ...        in dimension y

# Array initializations
T     = np.zeros((nx,   ny, ))
Cp    = np.zeros((nx,   ny, ))
dTedt = np.zeros((nx-2, ny-2))
qTx   = np.zeros((nx-1, ny-2))
qTy   = np.zeros((nx-2, ny-1))

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
x0    = coords[0]*(nx-2)*dx
y0    = coords[1]*(ny-2)*dy
Cp[:] = cp_min + np.reshape([  5*np.exp(-((x0 + ix*dx - lx/1.5)/1.0)**2 - ((y0 + iy*dy - ly/1.5)/1.0)**2) + 
                               5*np.exp(-((x0 + ix*dx - lx/1.5)/1.0)**2 - ((y0 + iy*dy - ly/3.0)/1.0)**2) for ix in range(nx) for iy in range(ny)], (nx,ny))
T[:]  =          np.reshape([100*np.exp(-((x0 + ix*dx - lx/3.0)/2.0)**2 - ((y0 + iy*dy - ly/2.0)/2.0)**2) +
                              50*np.exp(-((x0 + ix*dx - lx/1.5)/2.0)**2 - ((y0 + iy*dy - ly/2.0)/2.0)**2) for ix in range(nx) for iy in range(ny)], (nx,ny))

# ADIOS2 
# (size and start of the local and global problem)
nxy_nohalo   = [nx-2, ny-2]                                        # In ADIOS2 slang: count
nxy_g_nohalo = [nx_g-2, ny_g-2]                                    # ...              shape
start        = [coords[0]*nxy_nohalo[0], coords[1]*nxy_nohalo[1]]  # ...              start
T_nohalo     = np.zeros(nxy_nohalo)                                # Prealocate array for writing temperature
# (intialize ADIOS2, io, engine and define the variable temperature)
adios  = adios2.ADIOS(configFile="adios2.xml", comm=comm)
io     = adios.DeclareIO("writerIO")
T_id   = io.DefineVariable("temperature", T, nxy_g_nohalo, start, nxy_nohalo, adios2.ConstantDims)
engine = io.Open("diffusion2D.bp", adios2.Mode.Write)

# Time loop
nsteps = 50                                                      # Number of times data is written during the simulation
dt     = min(dx,dy)**2*cp_min/lam/4.1                            # Time step for the 2D Heat diffusion
t      = 0                                                       # Initialize physical time
tic    = time.time()                                             # Start measuring wall time
for it in range(nt):
    if it % (nt/nsteps) == 0:                                    # Write data only nsteps times
        T_nohalo[:] = T[1:-1, 1:-1]                              # Copy data removing the halo
        engine.BeginStep()                                       # Begin ADIOS2 write step
        engine.Put(T_id, T_nohalo)                               # Add T (without halo) to variables for writing
        engine.EndStep()                                         # End ADIOS2 write step (includes normally writing)
        print('Time step ' + str(it) + '...')
    qTx[:]       = -lam*np.diff(T[:,1:-1],axis=0)/dx             # Fourier's law of heat conduction: q_x   = -λ δT/δx
    qTy[:]       = -lam*np.diff(T[1:-1,:],axis=1)/dy             # ...                               q_y   = -λ δT/δy
    dTedt[:]     = 1.0/Cp[1:-1,1:-1]*(-np.diff(qTx,axis=0)/dx -  # Conservation of energy:           δT/δt = 1/cₚ(-δq_x/δx
                                       np.diff(qTy,axis=1)/dy)   #                                               - δq_y/dy)
    T[1:-1,1:-1] = T[1:-1,1:-1] + dt*dTedt                       # Update of temperature             T_new = T_old + δT/δt
    t            = t + dt                                        # Elapsed physical time
    update_halo(T, neighbors_x, neighbors_y)                     # Update the halo of T

engine.Close()
print( "time : {0:.8f}".format( time.time()-tic) )
print( 'Min. temperature %2.2e'%(np.min(T)) )
print( 'Max. temperature %2.2e'%(np.max(T)) )
