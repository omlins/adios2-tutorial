import numpy as np
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import adios2

def update_halo(A, neighbors_x, neighbors_y):
	if neighbors_x[0] >= 0: # MPI_PROC_NULL?
		sendbuf = np.copy(A[1,:]).flatten();
		recvbuf = np.zeros(A.shape[1]);
		comm.Send(sendbuf, neighbors_x[0], 0)
		comm.Recv(recvbuf,  neighbors_x[0], 1)
		A[0,:] = recvbuf
	if neighbors_x[1] >= 0: # MPI_PROC_NULL?
		sendbuf = np.copy(A[-2,:]).flatten();
		recvbuf = np.zeros(A.shape[1]);
		comm.Send(sendbuf, neighbors_x[1], 1)
		comm.Recv(recvbuf, neighbors_x[1], 0)
		A[-1,:] = recvbuf
	if neighbors_y[0] >= 0: # MPI_PROC_NULL?
		sendbuf = np.copy(A[:,1]).flatten();
		recvbuf = np.zeros(A.shape[0]);
		comm.Send(sendbuf, neighbors_y[0], 2)
		comm.Recv(recvbuf,  neighbors_y[0], 3)
		A[:,0] = recvbuf
	if neighbors_y[1] >= 0: # MPI_PROC_NULL?
		sendbuf = np.copy(A[:,-2]).flatten();
		recvbuf = np.zeros(A.shape[0]);
		comm.Send(sendbuf, neighbors_y[1], 3)
		comm.Recv(recvbuf, neighbors_y[1], 2)
		A[:,-1] = recvbuf

# MPI
nprocs = MPI.COMM_WORLD.Get_size()
dims   = MPI.Compute_dims(nprocs, [0,0])
comm   = MPI.COMM_WORLD.Create_cart(dims)
me     = comm.Get_rank()
coords = comm.Get_coords(me)
neighbors_x = comm.Shift(direction = 0,disp=1)
neighbors_y = comm.Shift(direction = 1,disp=1)

# Physics
lam        = 1.0;                             # Thermal conductivity
cp_min     = 1.0;                             # Minimal heat capacity
lx, ly     = 10.0, 10.0;                      # Length of computational domain in dimension x and y

# Numerics
nx, ny     = 128, 128;                        # Number of gridpoints in dimensions x and y
nt         = 10000;                           # Number of time steps
nx_g       = dims[0]*(nx-2) + 2
ny_g       = dims[1]*(ny-2) + 2
dx         = lx/(nx_g-1);                     # Space step in dimension x
dy         = ly/(ny_g-1);                     # ...        in dimension y

# Array initializations
T     = np.zeros((nx,   ny, ));
Cp    = np.zeros((nx,   ny, ));
dTedt = np.zeros((nx-2, ny-2));
qTx   = np.zeros((nx-1, ny-2));
qTy   = np.zeros((nx-2, ny-1));

#TODO: replace below - attention with dx
#x0      = coords[0]*(nx-2)*dx
#y0      = coords[1]*(ny-2)*dy
# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Cp[:] = cp_min + np.reshape([  5*np.exp(-(((coords[0]*(nx-2)+ix)*dx - lx/1.5)/1)**2 - (((coords[1]*(ny-2)+iy)*dy - ly/1.5)/1)**2) + 
	                           5*np.exp(-(((coords[0]*(nx-2)+ix)*dx - lx/1.5)/1)**2 - (((coords[1]*(ny-2)+iy)*dy - ly/3.0)/1)**2) for ix in range(nx) for iy in range(ny)], (nx,ny))
T[:]  =          np.reshape([100*np.exp(-(((coords[0]*(nx-2)+ix)*dx - lx/3.0)/2)**2 - (((coords[1]*(ny-2)+iy)*dy - ly/2.0)/2)**2) +
                              50*np.exp(-(((coords[0]*(nx-2)+ix)*dx - lx/1.5)/2)**2 - (((coords[1]*(ny-2)+iy)*dy - ly/2.0)/2)**2) for ix in range(nx) for iy in range(ny)], (nx,ny))

# ADIOS
# (Size and start of the local and global problem (in ADIOS2 slang: nxy: count; nxy_g: shape; start: start)
nxy_nohalo   = [nx-2, ny-2]
nxy_g_nohalo = [nx_g-2, ny_g-2] 
start        = [coords[0]*nxy_nohalo[0], coords[1]*nxy_nohalo[1]]
T_nohalo     = np.zeros(nxy_nohalo)
# TODO: see if first engine, then T_id works and makes more sense.

adios  = adios2.ADIOS(configFile="adios2.xml", comm=comm)
io     = adios.DeclareIO("writerIO")
T_id   = io.DefineVariable("temperature", T, nxy_g_nohalo, start, nxy_nohalo, adios2.ConstantDims)
engine = io.Open("diffusion2D.bp", adios2.Mode.Write)
nsteps = 50

# Time loop
tic = time.time()
dt  = min(dx,dy)**2*cp_min/lam/4.1;                               # Time step for the 3D Heat diffusion
t   = 0;
for it in range(nt):
	if it%(nt/nsteps) == 0:                                      # Write data only nsteps times
		T_nohalo[:] = T[1:-1, 1:-1]                              # Copy data removing the halo.
		engine.BeginStep()                                       # Begin ADIOS write step.
		engine.Put(T_id, T_nohalo)                               # Add T to variables for writing
		engine.EndStep()                                         # End ADIOS write step (includes normally writing).
		print('Time step ' + str(it) + '...')
	qTx[:]       = -lam*np.diff(T[:,1:-1],axis=0)/dx;            # Fourier's law of heat conduction: q_x   = -λ δT/δx
	qTy[:]       = -lam*np.diff(T[1:-1,:],axis=1)/dy;            # ...                               q_y   = -λ δT/δy
	dTedt[:]     = 1.0/Cp[1:-1,1:-1]*(-np.diff(qTx,axis=0)/dx - 
									   np.diff(qTy,axis=1)/dy);  # Conservation of energy:           δT/δt = 1/cₚ (-δq_x/δx - δq_y/dy)
	T[1:-1,1:-1] = T[1:-1,1:-1] + dt*dTedt;                      # Update of temperature             T_new = T_old + δT/δt
	t            = t + dt;                                       # Elapsed physical time
	update_halo(T, neighbors_x, neighbors_y)                     # Update the halo of T

engine.Close()
print( "time : {0:.8f}".format( time.time()-tic) )
print( 'Min. temperature %2.2e'%(np.min(T)) )
print( 'Max. temperature %2.2e'%(np.max(T)) )

# plt.figure()
# plt.title('Temperature at time %2.1f s'%(t))
# plt.contourf(T, 256, cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()
