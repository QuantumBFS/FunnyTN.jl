const ↑ = :upbond
const ↓ = :downbond
const → = :lastbond
const ← = :firstbond

getindex(arr::Tensor, ::Type{Val{:firstbond}}) = Leg(arr, 1)
getindex(arr::Tensor{T, N}, ::Type{Val{:lastbond}}) where {T, N} = Leg(arr, N)
getindex(arr::Tensor, s::Symbol) = getindex(arr, Val{s})

############### contraction #################
∾(l1::Leg, l2::Leg) = glue(l1, l2)
∘(ts1::Tensor, ts2::Tensor) = chain_tensors(ts1, ts2)
