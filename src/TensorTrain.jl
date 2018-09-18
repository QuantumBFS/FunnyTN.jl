abstract type TensorTrain{T, TT}<:AbstractTN{T, TT} end
@forward TensorTrain.tensors getindex, lastindex, setindex!, iterate, length, eltype, eachindex

# List Behavior.
push!(c::TensorTrain, val::AbstractMatrix) = (push!(c.tensors, val); c)
append!(c::TensorTrain, list) = (append!(c |> tensors, list); c)
prepend!(c::TensorTrain, list) = (prepend!(c |> tensors, list); c)
insert!(c::TensorTrain, key, val) = (insert!(c |> tensors, key, val); c)

bond(tt::TensorTrain, l::Int) = Bond(rlink(tt[l]), llink(tt[l+1]))
bondsize(tt::TensorTrain, l::Int) = size(tt[l], ndims(tt))
nsite(mps::TensorTrain) = length(mps |> tensors)
