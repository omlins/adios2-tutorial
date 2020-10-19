import numpy as np
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import adios2
# Physics
lx      = 1;
ly      = 1;
amp_T   = 100;                               # amplitude
sig     = 0.1;                               # bandwidth
k0      = 3;                                 # conductivity
k0_i    = 0.1;                               # conductivity
rho     = 3000;                              # density
c       = 1000;                              # thermal capacity
# Numerics
nx      = 256;                               # x resol
ny      = nx;                                # y resol
nt      = 10000;                             # number of time steps
# Pre-calculation
dx      = lx/(nx-1);                         # cell size
dy      = ly/(nx-1);                         # cell size
dt      = min(dx,dy)**2/(k0/rho/c)/4.2;      # time step
xc      = np.linspace(0,lx,nx);              # x coord
yc      = np.linspace(0,ly,ny);              # y coord
xc2,yc2 = np.meshgrid(xc,yc);                # coord grid
t       = 0;                                 # time init
K       = k0*np.ones((nx,ny));               # spatialy variable k
# Initial condition
T            = amp_T*np.exp( -(xc2-lx/2)**2/2/sig**2 -(yc2-ly/2)**2/2/sig**2 ); # T ini
locations    = np.logical_and(xc2<0, yc2>0)
K[locations] = k0_i;
# MPI
nprocs = MPI.COMM_WORLD.Get_size()
dims   = MPI.Compute_dims(nprocs, [0,0])
comm   = MPI.COMM_WORLD.Create_cart(dims)
me     = comm.Get_rank()
coords = comm.Get_coords(me)
# ADIOS
# (Size and start of the local and global problem (in ADIOS2 slang: nxy: count; nxy_global: shape; start: start)
nxy        = [ny, nx]
nxy_global = [dims[0]*ny, dims[1]*nx] # Note: in a real-world app, the halo must be handled in addition.
start      = [coords[0]*nx, coords[1]*ny]
# TODO: see if first engine, then T_id works and makes more sense.

adios  = adios2.ADIOS(configFile="adios2.xml", comm=comm)
io     = adios.DeclareIO("writerIO")
#DefineVariable( shape: List[int] = [], start: List[int] = [], count: List[int] = [], isConstantDims: bool = False) -> adios2::py11::Variable

T_id   = io.DefineVariable("temperature", T, nxy_global, start, nxy, adios2.ConstantDims)
engine = io.Open("diffusion2D.bp", adios2.Mode.Write)
nsteps = 50

# Time steps
tic = time.time()
for it in range(nt):
	if it%(nt/nsteps) == 0:
	    engine.BeginStep()
	    engine.Put(T_id, T)
	    engine.EndStep()
	    print('Time step ' + str(it) + '...')
	t            = t + dt;
	Kx           = 0.5*(K[1:,1:-1] + K[:-1,1:-1]);
	Ky           = 0.5*(K[1:-1,1:] + K[1:-1,:-1]);
	qTx          = -Kx*np.diff(T[:,1:-1],axis=0)/dx;
	qTy          = -Ky*np.diff(T[1:-1,:],axis=1)/dy;
	T[1:-1,1:-1] = T[1:-1,1:-1] - dt/rho/c*( np.diff(qTx,axis=0)/dx + np.diff(qTy,axis=1)/dy );

engine.Close()
print( "time : {0:.8f}".format( time.time()-tic) )
print( 'Min. temperature %2.2e'%(np.min(T)) )
print( 'Max. temperature %2.2e'%(np.max(T)) )

# plt.figure()
# plt.title('Temperature at time %2.1f s'%(t))
# plt.contourf(xc2, yc2, T, 256, cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()
