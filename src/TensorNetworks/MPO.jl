"""MPO Tensor"""
struct MPOTensor{T, AT<:DenseArray{T, 4}} <: Tensor{T, 4}
    data::AT
    MPOTensor(ts::AbstractArray{T, 4}) where T = new{T, typeof(ts)}(ts)
end

getindex(mt::MPOTensor, ::LegIndex{:up}) = 2
getindex(mt::MPOTensor, ::LegIndex{:down}) = 3

"""
    MPO{T} <: TensorTrain

Matrix Product Operator.

We use the following convention to number legs:
    2
    |
 1--A--4
    |
    3

llink -> 1
ulink -> 2
dlink -> 3
rlink -> 4
"""
mutable struct MPO{BC, T} <: MPSO{BC, T, 4, MPOTensor{T}}
    tensors::Vector{MPOTensor{T}}
    S::Vector{T}
    l::Int
    MPO{BC}(tensors::Vector{MPOTensor{T}}, S::Vector{T}, l::Int) where {BC, T} = new{BC, T}(tensors, S, l)
    MPO{BC}(tensors::Vector{MPOTensor{T}}, p::Pair{Int, Vector{T}}=0=>T[1]) where {T, BC} = MPO{BC}(tensors, p.second, p.first)
    MPO(tensors::Vector{MPOTensor{T}}, p::Pair{Int, Vector{T}}=0=>T[1]) where T = MPO{:open}(tensors, p)
end

tensors(mpo::MPO) = mpo.tensors
