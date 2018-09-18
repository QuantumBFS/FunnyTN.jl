"""MPO Tensor"""
struct MPOTensor{T, AT<:DenseArray{T, 4}} <: Tensor{T, 4}
    data::AT
    MPOTensor(ts::AbstractArray{T, 4}) where T = new{T, typeof(ts)}(ts)
end


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
struct MPO{T} <: TensorTrain{T, MPOTensor{T}}
    tensors::Vector{MPOTensor{T}}
    S::Vector{T}
    l::Int
    MPO(tensors::Vector{MPOTensor{T}}, S::Vector{T}, l::Int) where T = new{T}(tensors, S, l)
end

mps(tensors::Vector{MPOTensor{T}}, p::Pair{Int, Vector{T}}) where T = MPO(tensors, p.second, p.first)
mps(tensors::Vector{MPOTensor{T}}) where T = MPO(tensors, 0=>T[1])

tensors(mpo::MPO) = mpo.tensors
