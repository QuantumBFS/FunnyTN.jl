"""MPO Tensor"""
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
"""
mutable struct MPO{BC, T, TT} <: MPSO{BC, T, 4, TT}
    tensors::Vector{TT}
    l::Int
    MPO{BC}(tensors::Vector{TT}, l::Int=1) where {BC, T, TT<:MPOTensor{T}} = new{BC, T, TT}(tensors, l)
end

MPO(tensors::Vector{TT}, l::Int=1) where {T, TT<:MPOTensor{T}} = MPO{:open}(tensors, l)
tensors(mpo::MPO) = mpo.tensors

rand_mpo(nbit::Int; b::Int=5, nflavor::Int=2) = rand_mpo(ComplexF64, nbit, b=b, nflavor=nflavor)
function rand_mpo(::Type{T}, nbit::Int; b::Int=5, nflavor::Int=2) where T
    bi_last = 1
    mpo = MPO(Array{T, 4}[])
    for i=1:nbit
        bi = min(b, nflavor^(2*min(i, nbit-i)))
        push!(mpo, randn(bi_last, nflavor, nflavor, bi))
        bi_last = bi
    end
    mpo
end

function Base.Matrix(mpo::MPO{:open, T}) where T
    res = dropdims(mpo[1], dims=1)
    for i=2:nsite(mpo)
        res = res[→] |> absorb_mpo(mpo[i])
    end
    dropdims(res, dims=3)
end

function show(io::IO, mpo::MPO)
    print(io, "MPO($(length(mpo)))  ", size(mpo[1], 1), join(["-$(i==mpo.l ? "■" : "□")-$(size(t, 4))" for (i, t) in enumerate(mpo)], ""))
end
