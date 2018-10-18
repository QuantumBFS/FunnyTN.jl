module Tensors

using TensorOperations, LinearMaps, Lazy, LinearAlgebra
import Base: push!, append!, prepend!, length, parent, getindex, size, insert!, setindex!, iterate, eltype, eachindex, lastindex, convert, vec, show
import Base: +, -, *, /, ∘

export log2i, nqubits, all_equivalent, assert_samesize
export LegIndex, Tensor, MPSTensor, MPOTensor, TMatrix, Leg, Bond, lastleg, firstleg, axismap, axis
export glue, mulaxis!, chain_tensors, mulaxis

# structure related tensor APIs
export bra_ket_prod, t_bra_ket_prod, x_bra_ket_prod, mpo_ket_prod, tt_dadd

export ∾, ⧷,↑, ↓, ←, →

include("Core.jl")
include("Leg.jl")
include("linalg.jl")
include("contractions.jl")
include("interfaces.jl")

end
