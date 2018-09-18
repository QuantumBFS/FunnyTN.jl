module FunnyTN

using TensorOperations, LinearMaps, Lazy, LinearAlgebra
import Base: push!, append!, prepend!, length, parent, getindex, size, insert!, setindex!, iterate, eltype, eachindex, lastindex, convert

export AbstractTN, TensorTrain, MPSTensor, MPS, MPO, MPOTensor, Leg, Bond
export decompose, vec2mps, statevec
export ∾, ⧷,↑, ↓, ←, →

include("Core.jl")
include("Leg.jl")
include("TensorTrain.jl")
include("MPS.jl")
include("MPO.jl")
include("linalg.jl")
include("interfaces.jl")

end
