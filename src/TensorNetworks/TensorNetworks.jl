module TensorNetworks

using TensorOperations, LinearMaps, Lazy, LinearAlgebra
using ..Tensors

import Base: push!, append!, prepend!, length, parent, getindex, size, insert!, setindex!, iterate, eltype, eachindex, lastindex, convert, vec, show, copy, adjoint
import Base: +, -, *, /, sum, >>, <<
import LinearAlgebra: rmul!, lmul!, kron, normalize!, norm

export AbstractTN, TensorTrain, MPSO, tensors
export assert_boundary_match, assert_chainable, assert_valid, assert_canonical
export MPS, bondsizes, bondsize, rand_mps, bcond, nsite, hsize, hgetindex, canomove!, cloc, ccenter
export MPO, mpo, rand_mpo, nflavor
export svdtrunc, decompose, vec2mps, compress!, recanonicalize!, inner_product, tmatrix
export CanonicalityError

include("Core.jl")
include("TensorTrain.jl")
include("MPS.jl")
include("MPO.jl")
include("linalg.jl")
include("interfaces.jl")

end
