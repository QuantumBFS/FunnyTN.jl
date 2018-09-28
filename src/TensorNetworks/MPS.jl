"""MPS Tensor"""
const MPSTensor{T} = AbstractArray{T, 3}

mapaxis(mt::MPSTensor, ::LegIndex{:up}) = 2
mapaxis(mt::MPSTensor, ::LegIndex{:down}) = 2

"""
    MPS{T, BC, TT<:MPSTensor{T}} <: TensorTrain{T, TT}
    MPS{BC}(tensors::Vector{TT}, l::Int) -> MPS

matrix product state, BC is the boundary condition, l is the canonical center (0 if not needed).
"""
mutable struct MPS{BC, T, TT<:MPSTensor{T}} <: MPSO{BC, T, 3, TT}
    tensors::Vector{TT}
    l::Int
    function MPS{BC}(tensors::Vector{TT}, l::Int) where {BC, T, TT<:MPSTensor{T}}
        new{BC, T, TT}(tensors, l)
    end
    MPS(tensors::Vector, l::Int=1) = MPS{:open}(tensors, l)
end

function assert_valid(mps::MPS{BC}) where BC
    tss = mps |> tensors
    l = cloc(mps)
    assert_boundary_match(tss, BC)
    assert_samesize(tss, 2)
    assert_chainable(tss)
    true
end

function assert_canonical(mps::MPS)
    for i in 1:mps.l-1
        U = reshape(m, :, size(m, 3))
        U'*U == I || throw(CanonicalityError("MPS canonicality error!"))
    end
    for i in 1:mps.l+1:nsite(mps)
        V = reshape(m, size(m, 1), :)
        V*V' == I || throw(CanonicalityError("MPS canonicality error!"))
    end
end

tensors(mps::MPS) = mps.tensors
copy(m::MPS) = MPS{bcond(m)}([t for t in m.tensors], m.l)
deepcopy(m::MPS) = MPS{bcond(m)}([t |> copy for t in m.tensors], m.l)

"""
    rand_mps([::Type], nflavor::Int, bond_dims::Vector{Int}) -> MPS

Random matrix product state.
"""
rand_mps(::Type{T}, nflavor::Int, bond_dims::Vector{Int}; l=1) where T = MPS([randn(T, bond_dims[i], nflavor, bond_dims[i+1]) for i = 1:length(bond_dims)-1], l)
rand_mps(nflavor::Int, bond_dims::Vector{Int}; l=1) = rand_mps(ComplexF64, nflavor, bond_dims, l=l)

cloc(mps::MPS) = mps.l
ccenter(mps::MPS) = mps[mps.l]
nflavor(mps::MPS) = size(first(mps), 2)
adjoint(ket::MPS) = Adjoint(ket)
bcond(mps::MPS{BC, T}) where {T, BC} = BC

function show(io::IO, mps::MPS)
    print(io, "MPS($(length(mps)))  ", size(mps[1], 1), join(["-[$(size(t, 2))$(i==mps.l ? "*" : "")]-$(size(t, 3))" for (i, t) in enumerate(mps)], ""))
end
function show(io::IO, bra::Adjoint{<:Any, <:MPS})
    mps = bra |> parent
    print(io, "Bra($(length(mps)))  ", size(mps[1], 1), join(["-[$(size(t, 2))$(i==mps.l ? "*" : "")]-$(size(t, 3))" for (i, t) in enumerate(mps)], ""))
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
    reduce(*, [selectdim(t, 2, ind) for (ind, t) in zip(inds, tensors(mps))])[]
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

@forward Bra.parent tensors, nflavor, bondsizes, bondsize, hsize
hgetindex(bra::Bra, ind) = hgetindex(bra |> parent) |> conj
copy(m::Bra) = adjoint(parent(m) |> copy)
function show(io::IO, mps::Bra)
    print(io, "Bra($(length(mps)))  ", size(mps[1], 1),(0==mps.l ? "*" : ""), join(["-[$(size(t, 2))]-$(size(t, 3))$(i==mps.l ? "*" : "")" for (i, t) in enumerate(mps)], ""))
end
=#
