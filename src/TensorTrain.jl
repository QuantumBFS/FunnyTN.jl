abstract type TensorTrain{T}<:AbstractTN{T} end

# List Behavior.
@forward TensorTrain.data Base.getindex, Compat.lastindex, Base.setindex!, Base.start, Base.next, Base.done, Base.length, Base.eltype, Base.eachindex, Base.insert!
push!(c::TensorTrain, val::AbstractBlock) = (push!(c.data, val); c)
append!(c::TensorTrain, list) = (append!(c.data, list); c)
prepend!(c::TensorTrain, list) = (prepend!(c.data, list); c)

"""
    bond(tt::TensorTrain, l::Int)

l-th bound of tensor train.
"""
bond(tt::TensorTrain, l::Int) = Bond(rlink(tt[l]), llink(tt[l+1]))

"""
    tensors(tt::TensorTrain) -> Tensor

Get the list tensors.
"""
function tensors end

"""
    llink(tt::Tensor)

left leg of a tensor.
"""
function llink end

"""
    rlink(tt::Tensor)

right leg of a tensor.
"""
function rlink end
