# ADIOS2Tutorial.jl

Adapted from the Python code [here](https://github.com/omlins/adios2-tutorial).

Run these two commands simultaneoulsy (in different terminals):

```sh
cd example-julia
julia --project=@. -e 'using ADIOS2Tutorial; ADIOS2Tutorial.write()'
```

```sh
cd example-julia
julia --project=@. -e 'using ADIOS2Tutorial; ADIOS2Tutorial.read()'
```
