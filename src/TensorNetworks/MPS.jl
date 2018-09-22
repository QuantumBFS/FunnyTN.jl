"""MPS Tensor"""
const MPSTensor{T} = AbstractArray{T, 3}

getindex(mt::MPSTensor, ::Type{Val{:upbond}}) = Leg(mt, 2)
getindex(mt::MPSTensor, inds::Int...) = getindex(mt.data, inds...)

struct MPS{T, TT<:AbstractArray{T, 3}} <: TensorTrain{T, TT}
    tensors::Vector{TT}
    S::Vector{T}
    l::Int
    MPS(tensors::Vector{TT}, S::Vector{T}, l::Int) where {T, TT<:AbstractArray{T, 3}} = new{T, TT}(tensors, S, l)
end

mps(tensors::Vector{TT}, p::Pair{Int, Vector{T}}) where {T, TT<:MPSTensor{T}} = MPS(tensors, p.second, p.first)
mps(tensors::Vector{TT}) where {T, TT<:MPSTensor{T}} = mps(tensors, 0=>T[1])

tensors(mps::MPS) = mps.tensors
rand_mps(::Type{T}, nflavor::Int, bond_dims::Vector{Int}) where T = mps([randn(T, bond_dims[i], nflavor, bond_dims[i+1]) for i = 1:length(bond_dims)-1])
rand_mps(nflavor::Int, bond_dims::Vector{Int}) = rand_mps(ComplexF64, nflavor, bond_dims)

singular_values(mps::MPS) = mps.S.diag
l_canonical(mps::MPS) = mps.l

function show(io::IO, mps::MPS)
    print(io, "MPS($(length(mps)))  ", size(mps[1], 1),(0==mps.l ? "ˢ" : ""), join(["-[$(size(t, 2))]-$(size(t, 3))$(i==mps.l ? "*" : "")" for (i, t) in enumerate(mps)], ""))
end
