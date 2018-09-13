struct MPSTensor{T, AT<:StridedArray{T, 3}} <: StridedArray{T, 3}
    data::AT
end

@forward MPSTensor.data getindex, setindex!, size

tensors(mps::MPS) = mps.data
llink(tt::MPSTensor) = Leg(tt, 1)
rlink(tt::MPSTensor) = Leg(tt, 3)

"""
    slink(tt::MPSTensor)

physical leg of an MPS Tensor.
"""
slink(tt::MPSTensor) = Leg(tt, 2)

"""
    MPS{T} <: TensorTrain{T}

Matrix Product State

We use the following convention for numbering legs:
 1--A--3
    |
    2

llink -> 1
slink -> 2
rlink -> 3
"""
struct MPS{T, TT} <: TensorTrain{T, TT<:MPSTensor{T}}
    data::Vector{TT}
    n::Int
end

nsite(mps::MPS) = mps.n
