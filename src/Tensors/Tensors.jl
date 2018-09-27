module Tensors

using TensorOperations, LinearMaps, Lazy, LinearAlgebra
import Base: push!, append!, prepend!, length, parent, getindex, size, insert!, setindex!, iterate, eltype, eachindex, lastindex, convert, vec, show
import Base: +, -, *, /, ∘

export log2i, nqubits, all_equivalent, assert_samesize
export Tensor, Leg, Bond, lastleg, firstleg
export glue, mulaxis!, chain_tensors
export ∾, ⧷,↑, ↓, ←, →

include("Core.jl")
include("Leg.jl")
include("linalg.jl")
include("interfaces.jl")

end
