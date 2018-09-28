"""MPS Tensor"""
const MPSTensor{T} = AbstractArray{T, 3}

mapaxis(mt::MPSTensor, ::LegIndex{:up}) = 2
mapaxis(mt::MPSTensor, ::LegIndex{:down}) = 2

"""
    MPS{T, BC, TT<:MPSTensor{T}} <: TensorTrain{T, TT}
    MPS{BC}(tensors::Vector{TT}, S::Vector{T}, l::Int) -> MPS
    MPS{BC::Symbol}(tensors::Vector{TT}, [p::Pair{Int, Vector{T}}]) -> MPS

matrix product state, BC is the boundary condition.
"""
mutable struct MPS{BC, T, TT<:MPSTensor{T}} <: MPSO{BC, T, 3, TT}
    tensors::Vector{TT}
    S::Vector{T}
    l::Int
    function MPS{BC}(tensors::Vector{TT}, S::Vector{T}, l::Int) where {BC, T, TT<:MPSTensor{T}}
        new{BC, T, TT}(tensors, S, l)
    end
    MPS{BC}(tensors::Vector{TT}, p::Pair{Int, Vector{T}}=0=>T[1]) where {BC, T, TT<:MPSTensor{T}} = MPS{BC}(tensors, p.second, p.first)
    MPS(tensors::Vector{TT}, p::Pair{Int, Vector{T}}=0=>T[1]) where {T, TT<:MPSTensor{T}} = MPS{:open}(tensors, p)
end

function assert_valid(mps::MPS{BC}) where BC
    tss = mps |> tensors
    l = l_canonical(mps)
    S = singular_values(mps)
    assert_boundary_match(tss, BC)
    assert_samesize(tss, 2)
    assert_chainable(tss)
    if !((l > 0 && size(tss[l], 3) == length(S)) || (l==0 && size(tss[1], 1)==length(S)))
        throw(DimensionMismatch("canonical position error, or size of S-matrix dimension error."))
    end
    true
end

tensors(mps::MPS) = mps.tensors
function tensors_withS(mps::MPS)
    res = [t for t in mps.tensors]
    l = mps |> l_canonical
    if l == 0
        res[1] = mulaxis(res[1], 1, mps |> singular_values)
    else
        res[l] = mulaxis(res[l], 3, mps |> singular_values)
    end
    res
end
copy(m::MPS) = MPS{bcond(m)}([t for t in m.tensors], copy(m.S), m.l)
deepcopy(m::MPS) = MPS{bcond(m)}([t |> copy for t in m.tensors], copy(m.S), m.l)

"""
    rand_mps([::Type], nflavor::Int, bond_dims::Vector{Int}) -> MPS

Random matrix product state.
"""
rand_mps(::Type{T}, nflavor::Int, bond_dims::Vector{Int}; l=0) where T = MPS([randn(T, bond_dims[i], nflavor, bond_dims[i+1]) for i = 1:length(bond_dims)-1], l=>randn(T, bond_dims[l+1]))
rand_mps(nflavor::Int, bond_dims::Vector{Int}; l=0) = rand_mps(ComplexF64, nflavor, bond_dims, l=l)

singular_values(mps::MPS) = mps.S
l_canonical(mps::MPS) = mps.l
nflavor(mps::MPS) = size(first(mps), 2)
adjoint(ket::MPS) = Adjoint(ket)
bcond(mps::MPS{BC, T}) where {T, BC} = BC

function show(io::IO, mps::MPS)
    print(io, "MPS($(length(mps)))  ", size(mps[1], 1),(0==mps.l ? "*" : ""), join(["-[$(size(t, 2))]-$(size(t, 3))$(i==mps.l ? "*" : "")" for (i, t) in enumerate(mps)], ""))
end

"""
    hsize(mps::MPS, [i::Int])

size of hilbert space.
"""
hsize(mps::MPS) = (nflavor(mps)^nsite(mps),)
hsize(mps::MPS, i::Int) = i==1 ? nflavor(mps)^nsite(mps) : throw(DimensionMismatch("MPS is one dimensional!"))
"""
    hgetindex(mps::MPS, inds) -> Number

Returns the amplitude of specific configuration in hilbert space, ind can be either integer or tuple.
"""
function hgetindex(mps::MPS, inds::Tuple)
    reduce(*, insert!(Any[selectdim(t, 2, ind) for (ind, t) in zip(inds, tensors(mps))], mps.l+1, Diagonal(mps.S)))[]
end
function hgetindex(mps::MPS, ind::Int)
    nflv = nflavor(mps)
    hgetindex(mps, CartesianIndices(Tuple(nflv for i=1:nsite(mps)))[ind].I)
end

#=
##################### Bra #####################
const Ket = MPS
const Bra{BC, T, TT} = Adjoint{T, <:MPS{BC, T, TT}}
const KetBra{BC, T, TT} = Union{Ket{BC, T, TT}, Bra{BC, T, TT}}

@forward Bra.parent tensors, l_canonical, nflavor, bondsizes, bondsize, hsize
hgetindex(bra::Bra, ind) = hgetindex(bra |> parent) |> conj
copy(m::Bra) = adjoint(parent(m) |> copy)
function show(io::IO, mps::Bra)
    print(io, "Bra($(length(mps)))  ", size(mps[1], 1),(0==mps.l ? "*" : ""), join(["-[$(size(t, 2))]-$(size(t, 3))$(i==mps.l ? "*" : "")" for (i, t) in enumerate(mps)], ""))
end
=#
