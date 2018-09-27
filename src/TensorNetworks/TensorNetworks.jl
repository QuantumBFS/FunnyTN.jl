module TensorNetworks

using TensorOperations, LinearMaps, Lazy, LinearAlgebra
using ..Tensors

import Base: push!, append!, prepend!, length, parent, getindex, size, insert!, setindex!, iterate, eltype, eachindex, lastindex, convert, vec, show, copy
import Base: +, -, *, /, sum, >>, <<
import LinearAlgebra: rmul!, lmul!, kron

export AbstractTN, TensorTrain, MPSO, tensors
export assert_boundary_match, assert_chainable, assert_valid
export MPSTensor, MPS, bondsizes, bondsize, rand_mps, singular_values, l_canonical, bcond, nsite, hsize, hgetindex, canomove!
export MPO, MPOTensor, mpo, rand_mpo, nflavor
export svdtrunc, decompose, vec2mps, compress!, recanonicalize!

include("Core.jl")
include("TensorTrain.jl")
include("MPS.jl")
include("MPO.jl")
include("linalg.jl")
include("interfaces.jl")

end
