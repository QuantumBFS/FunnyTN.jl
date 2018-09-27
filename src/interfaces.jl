const ↑ = :upbond
const ↓ = :downbond
const → = :lastbond
const ← = :firstbond

getindex(arr::Tensor, ::Val{:firstbond}) = Leg(arr, 1)
getindex(arr::Tensor{T, N}, ::Val{:lastbond}) where {T, N} = Leg(arr, N)
getindex(arr::Tensor, s::Symbol) = getindex(arr, Val(s))

getindex(mt::Union{MPSTensor, MPOTensor}, ::Val{:upbond}) = Leg(mt, 2)
getindex(mt::MPOTensor, ::Val{:downbond}) = Leg(mt, 3)
getindex(mt::MPSTensor, inds::Int...) = getindex(mt.data, inds...)

############### contraction #################
∾(l1, l2) = glue(l1, l2)
