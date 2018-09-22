module TensorNetworks

using TensorOperations, LinearMaps, Lazy, LinearAlgebra
using ..Tensors

import Base: push!, append!, prepend!, length, parent, getindex, size, insert!, setindex!, iterate, eltype, eachindex, lastindex, convert, vec, show
import Base: +, -, *, /

export AbstractTN, TensorTrain, tensors
export MPSTensor, MPS, mps, bondsize, rand_mps
export MPO, MPOTensor, mpo, rand_mpo
export decompose, vec2mps

include("Core.jl")
include("TensorTrain.jl")
include("MPS.jl")
include("MPO.jl")
include("linalg.jl")
include("interfaces.jl")

end
