"""MPS Tensor"""
const MPSTensor{T} = AbstractArray{T, 3}

getindex(mt::MPSTensor, ::Type{Val{:upbond}}) = Leg(mt, 2)
getindex(mt::MPSTensor, inds::Int...) = getindex(mt.data, inds...)

"""
    MPS{T, BC, TT<:MPSTensor{T}} <: TensorTrain{T, TT}
    MPS{BC}(tensors::Vector{TT}, S::Vector{T}, l::Int) -> MPS
    MPS{BC::Symbol}(tensors::Vector{TT}, [p::Pair{Int, Vector{T}}]) -> MPS

matrix product state, BC is the boundary condition.
"""
struct MPS{BC, T, TT<:MPSTensor{T}} <: MPSO{BC, T, 3, TT}
    tensors::Vector{TT}
    S::Vector{T}
    l::Int
    function MPS{BC}(tensors::Vector{TT}, S::Vector{T}, l::Int) where {BC, T, TT<:MPSTensor{T}}
        assert_boundary_match(tensors, BC)
        assert_samesize(tensors, 2)
        assert_chainable(tensors)
        if !((l > 0 && size(tensors[l], 3) == length(S)) || (l==0 && size(tensors[1], 1)==length(S)))
            throw(DimensionMismatch("canonical position error, or size of S-matrix dimension error."))
        end
        new{BC, T, TT}(tensors, S, l)
    end
    MPS{BC}(tensors::Vector{TT}, p::Pair{Int, Vector{T}}=0=>T[1]) where {BC, T, TT<:MPSTensor{T}} = MPS{BC}(tensors, p.second, p.first)
    MPS(tensors::Vector{TT}, p::Pair{Int, Vector{T}}=0=>T[1]) where {T, TT<:MPSTensor{T}} = MPS{:open}(tensors, p)
end

tensors(mps::MPS) = mps.tensors
copy(m::MPS) = MPS{bcond(m)}([t for t in m.tensors], copy(m.S), m.l)

"""
    rand_mps([::Type], nflavor::Int, bond_dims::Vector{Int}) -> MPS

Random matrix product state.
"""
rand_mps(::Type{T}, nflavor::Int, bond_dims::Vector{Int}; l=0) where T = MPS([randn(T, bond_dims[i], nflavor, bond_dims[i+1]) for i = 1:length(bond_dims)-1], l=>randn(T, bond_dims[l+1]))
rand_mps(nflavor::Int, bond_dims::Vector{Int}; l=0) = rand_mps(ComplexF64, nflavor, bond_dims, l=l)

singular_values(mps::MPS) = mps.S
l_canonical(mps::MPS) = mps.l
nflavor(mps::MPS) = size(first(mps), 2)
bcond(mps::MPS{BC, T}) where {T, BC} = BC

function show(io::IO, mps::MPS)
    print(io, "MPS($(length(mps)))  ", size(mps[1], 1),(0==mps.l ? "ˢ" : ""), join(["-[$(size(t, 2))]-$(size(t, 3))$(i==mps.l ? "*" : "")" for (i, t) in enumerate(mps)], ""))
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
