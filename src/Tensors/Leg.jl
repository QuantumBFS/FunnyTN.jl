import Base: @_pure_meta
################### Legs ###########################
struct LegIndex{AXIS} end
LegIndex(x) = (@_pure_meta; LegIndex{x}())
const TensorOrI = Union{Tensor, UniformScaling}

struct Leg{AXIS, AT}
    ts::AT
    Leg{AXIS}(ts::AT) where {AXIS, AT<:TensorOrI} = new{AXIS, AT}(ts)
    Leg(ts::TensorOrI, axis::Int) = Leg{axis}(ts)
    Leg(ts::TensorOrI, axis::LegIndex{AXIS}) where AXIS = Leg{axismap(ts, axis)}(ts)
end
lastleg(ts::Tensor{T, N}) where {T, N} = Leg{N}(ts)
firstleg(ts::Tensor) = Leg{1}(ts)

parent(leg::Leg) = leg.ts
axis(leg::Leg{AXIS}) where AXIS = AXIS

length(leg::Leg{AXIS}) where AXIS = size(leg.ts, AXIS)
show(io::IO, leg::Leg{AXIS}) where AXIS = print(io, "$(leg.ts |> typeof)$(leg.ts |> size) ⧷ $AXIS")

getindex(arr::Tensor, axis::LegIndex{AXIS}) where AXIS = Leg(arr, axis)
getindex(arr::UniformScaling, axis::LegIndex{AXIS}) where AXIS = Leg(arr, axis)
getindex(arr::Union{Tensor, UniformScaling}, s::Symbol) = getindex(arr, LegIndex{s}())

axismap(ts::TensorOrI, ::LegIndex{X}) where X = X
axismap(ts::Tensor{T, N}, ::LegIndex{:last}) where {T, N} = N
axismap(ts::Tensor, ::LegIndex{:first}) = 1

axismap(ts::Union{MPSTensor, MPOTensor}, ::LegIndex{:up}) = 2
axismap(ts::MPOTensor, ::LegIndex{:down}) = 3
axismap(ts::Adjoint{<:Any, <:MPSTensor}, ::LegIndex{:down}) = 2

################## Bonds #######################
struct Bond{C, L<:Leg{C}}
    leg1::L
    leg2::L
end
