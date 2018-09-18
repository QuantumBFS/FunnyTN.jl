"""MPS Tensor"""
const MPSTensor{T} = AbstractArray{T, 3}

#struct MPSTensor{T, AT<:DenseArray{T, 3}} <: Tensor{T, 3}
#    data::AT
#    MPSTensor(ts::AbstractArray{T, 3}) where T = new{T, typeof(ts)}(ts)
#end
#@forward MPSTensor.data setindex!, size, eltype
#data(mt::MPSTensor) = mt.data
#convert(MPSTensor, da::DenseArray{T, 3}) where T = MPSTensor(da)

struct MPS{T, TT<:AbstractArray{T, 3}} <: TensorTrain{T, TT}
    tensors::Vector{TT}
    S::Vector{T}
    l::Int
    MPS(tensors::Vector{TT}, S::Vector{T}, l::Int) where {T, TT<:AbstractArray{T, 3}} = new{T, TT}(tensors, S, l)
end

mps(tensors::Vector{TT}, p::Pair{Int, Vector{T}}) where {T, TT<:MPSTensor{T}} = MPS(tensors, p.second, p.first)
mps(tensors::Vector{TT}) where {T, TT<:MPSTensor{T}} = MPS(tensors, 0=>T[1])

tensors(mps::MPS) = mps.tensors
singular_values(mps::MPS) = mps.S.diag
l_canonical(mps::MPS) = mps.l

function Base.show(io::IO, mps::MPS)
    print(io, "MPS($(length(mps)))  ", size(mps[1], 1),(0==mps.l ? "ˢ" : ""), join(["-[$(size(t, 2))]-$(size(t, 3))$(i==mps.l ? "*" : "")" for (i, t) in enumerate(mps)], ""))
end
