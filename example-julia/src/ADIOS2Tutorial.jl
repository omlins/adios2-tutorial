module ADIOS2Tutorial

using ADIOS2
using MPI

function MPI_Dims_create(nprocs, ndims)
    dims = zeros(Cint, ndims)
    MPI.Dims_create!(nprocs, ndims, dims)
    return dims
end

function update_halo(A::AbstractArray{Float64,2}, neighbors_x::NTuple{2,Int}, neighbors_y::NTuple{2,Int}, comm::MPI.Comm)
    if neighbors_x[1] >= 0      # MPI_PROC_NULL?
        sendbuf = np[2, :]
        recvbuf = zeros(size(A, 1))
        Sendrecv!(sendbuf, neighbors_x[1], 0, recvbuf, neighbors_x[1], 1, comm)
        A[1, :] .= recvbuf
    end
    if neighbors_x[2] >= 0      # MPI_PROC_NULL?
        sendbuf = np[end - 1, :]
        recvbuf = zeros(size(A, 1))
        Sendrecv!(sendbuf, neighbors_x[2], 1, recvbuf, neighbors_x[2], 0, comm)
        A[end, :] .= recvbuf
    end
    if neighbors_y[1] >= 0      # MPI_PROC_NULL?
        sendbuf = np[:, 2]
        recvbuf = zeros(size(A, 2))
        Sendrecv!(sendbuf, neighbors_y[1], 2, recvbuf, neighbors_y[1], 3, comm)
        A[:, 1] .= recvbuf
    end
    if neighbors_y[2] >= 0      # MPI_PROC_NULL?
        sendbuf = np[:, end - 1]
        recvbuf = zeros(size(A, 2))
        Sendrecv!(sendbuf, neighbors_y[2], 3, recvbuf, neighbors_y[2], 2, comm)
        A[:, end] .= recvbuf
    end
end

function write()
    MPI.Init()

    # MPI
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    dims = MPI_Dims_create(nprocs, 2)
    comm = MPI.Cart_create(MPI.COMM_WORLD, length(dims), dims, Cint[0, 0], true)
    me = MPI.Comm_rank(comm)
    coords = MPI.Cart_coords(comm)
    neighbors_x = MPI.Cart_shift(comm, 0, 1)
    neighbors_y = MPI.Cart_shift(comm, 1, 1)

    # Physics
    lam = 1.0           # Thermal conductivity
    cp_min = 1.0        # Minimal heat capacity
    lx, ly = 10.0, 10.0 # Length of computational domain in dimension x and y

    # Numerics
    nx, ny = 128, 128     # Number of gridpoints in dimensions x and y
    nt = 10000            # Number of time steps
    nx_g = dims[1] * (nx - 2) + 2 # Number of gridpoints of the global problem in dimension x
    ny_g = dims[2] * (ny - 2) + 2 # ...                                        in dimension y
    dx = lx / (nx_g - 1)          # Space step in dimension x
    dy = ly / (ny_g - 1)          # ...        in dimension y

    # Array initializations
    T = zeros(nx, ny)
    Cp = zeros(nx, ny)
    # dTedt = zeros(nx - 2, ny - 2)
    # qTx = zeros(nx - 1, ny - 2)
    # qTy = zeros(nx - 2, ny - 1)
    dTedt = zeros(nx, ny)

    # Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
    x0 = coords[1] * (nx - 2) * dx
    y0 = coords[2] * (ny - 2) * dy
    Cp = [cp_min +
          5 * exp(-((x0 + ix * dx - lx / 1.5) / 1.0)^2 - ((y0 + iy * dy - ly / 1.5) / 1.0)^2) +
          5 * exp(-((x0 + ix * dx - lx / 1.5) / 1.0)^2 - ((y0 + iy * dy - ly / 3.0) / 1.0)^2) for ix in 1:nx, iy in 1:ny]
    T = [100 * exp(-((x0 + ix * dx - lx / 3.0) / 2.0)^2 - ((y0 + iy * dy - ly / 2.0) / 2.0)^2) +
         50 * exp(-((x0 + ix * dx - lx / 1.5) / 2.0)^2 - ((y0 + iy * dy - ly / 2.0) / 2.0)^2) for ix in 1:nx, iy in 1:ny]

    # ADIOS2 
    # (size and start of the local and global problem)
    nxy_nohalo = (nx - 2, ny - 2)       # In ADIOS2 slang: count
    nxy_g_nohalo = (nx_g - 2, ny_g - 2) # ...              shape
    start = Tuple(coords) .* nxy_nohalo # ...              start
    T_nohalo = zeros(nxy_nohalo) # Prealocate array for writing temperature
    # (intialize ADIOS2, io, engine and define the variable temperature)
    adios = adios_init_mpi("adios2.xml", comm)   # Use the configurations defined in "adios2.xml"...
    io = declare_io(adios, "writerIO") # ... in the section "writerIO"
    T_id = define_variable(io, "temperature", eltype(T), nxy_g_nohalo, start, nxy_nohalo; constant_dims=true) # Define the variable "temperature"
    engine = open(io, "diffusion2D.bp", mode_write)

    # Time loop
    nsteps = 50 # Number of times data is written during the simulation
    dt = min(dx, dy)^2 * cp_min / lam / 4.1 # Time step for the 2D Heat diffusion
    t = 0                                   # Initialize physical time
    tic = time()                # Start measuring wall time
    for it in 1:nt
        if it % (nt ÷ nsteps) == 0 # Write data only nsteps times
            T_nohalo = T[2:(end - 1), 2:(end - 1)] # Copy data removing the halo
            begin_step(engine)  # Begin ADIOS2 write step
            put!(engine, T_id, T_nohalo) # Add T (without halo) to variables for writing
            end_step(engine) # End ADIOS2 write step (includes normally the actual writing of data)
            println("Time step $it...")
        end

        for j in 2:(ny - 1), i in 2:(nx - 1)
            qTxm = -lam * (T[i, j] - T[i - 1, j]) / dx # Fourier's law of heat conduction: q_x   = -λ δT/δx
            qTxp = -lam * (T[i + 1, j] - T[i, j]) / dx
            qTym = -lam * (T[i, j] - T[i, j - 1]) / dy # ...                               q_y   = -λ δT/δy
            qTyp = -lam * (T[i, j + 1] - T[i, j]) / dy

            dTedt[i, j] = 1.0 / Cp[i, j] * (-(qTxp - qTxm) / dx - (qTyp - qTym) / dy)# Conservation of energy: δT/δt = 1/cₚ(-δq_x/δx - δq_y/dy)
        end
        for j in 2:(ny - 1), i in 2:(nx - 1)
            T[i, j] += dt * dTedt[i, j] # Update of temperature             T_new = T_old + δT/δt
        end

        t += dt                 # Elapsed physical time
        update_halo(T, neighbors_x, neighbors_y, comm) # Update the halo of T
    end

    close(engine)
    println("time: $(time() - tic)")
    println("Min. temperature $(minimum(T))")
    println("Max. temperature $(maximum(T))")
    return
end

function read()
    adios = adios_init_serial("adios2.xml")
    io = declare_io(adios, "readerIO")
    @show engine_type(io)
    engine = open(io, "diffusion2D.bp", mode_read)
    @show type(engine)
    @show name.(inquire_all_variables(io))

    T = nothing
    nprocessed = 0
    while begin_step(engine, step_mode_read, 100.0) != step_status_end_of_stream
        T_id = inquire_variable(io, "temperature")
        if nprocessed == 0
            nxy_global = shape(T_id)
            nxy = count(T_id)
            T_type = type(T_id)
            T = zeros(T_type, nxy)
            println(nxy_global, nxy, T_type)
            sleep(4)
        end
        get(engine, T_id, T)
        end_step(engine)
        println(nprocessed, minimum(T), maximum(T))
        # IPython.display.clear_output(wait=True)
        # plt.title('Temperature at step ' + str(engine.CurrentStep()))
        # plt.contourf(T, 256, cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.show()
        nprocessed += 1
    end

    return
end

end
