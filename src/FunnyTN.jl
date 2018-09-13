using TensorOperations
using LinearMaps

import Base: push!, append!, prepend!, length, parent, getindex, setindex, size

export AbstractTN, TensorTrain, MPS

include("tensortrain.jl")
include("mps.jl")
include("mpo.jl")
