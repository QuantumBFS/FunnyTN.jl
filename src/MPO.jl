const MPOTensor{T} = AbstractArray{T, 4}

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
struct MPO{T} <: TensorTrain
    data::Vector{MPOTensor{T}}
end

tensors(mpo::MPO) = mpo.data
llink(tt::MPOTensor) = Leg(tt, 1)
rlink(tt::MPOTensor) = Leg(tt, 4)
"""
    ulink(tt::Tensor)

upside leg of an MPO.
"""
ulink(tt::MPOTensor) = Leg(tt, 2)

"""
    dlink(tt::Tensor)

downside leg of an MPO.
"""
dlink(tt::MPOTensor) = Leg(tt, 3)
