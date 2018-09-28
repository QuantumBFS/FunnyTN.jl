import Base: @_pure_meta
################### Legs ###########################
struct LegIndex{AXIS} end
LegIndex(x) = (@_pure_meta; LegIndex{x}())

struct Leg{AXIS, AT<:Tensor}
    ts::AT
    Leg{AXIS}(ts::AT) where {AXIS, AT<:Tensor} = new{AXIS, AT}(ts)
    Leg(ts::Tensor, axis::Int) = Leg{axis}(ts)
    Leg(ts::Tensor, axis::LegIndex{AXIS}) where AXIS = Leg{axismap(ts, axis)}(ts)
end
lastleg(ts::Tensor{T, N}) where {T, N} = Leg{N}(ts)
firstleg(ts::Tensor) = Leg{1}(ts)

parent(leg::Leg) = leg.ts
axis(leg::Leg{AXIS}) where AXIS = AXIS

length(leg::Leg{AXIS}) where AXIS = size(leg.ts, AXIS)
show(io::IO, leg::Leg{AXIS}) where AXIS = print(io, "$(leg.ts |> typeof)$(leg.ts |> size) ⧷ $AXIS")

axismap(ts::Tensor, ::LegIndex{X}) where X = X
axismap(ts::Tensor{T, N}, ::LegIndex{:last}) where {T, N} = N
axismap(ts::Tensor, ::LegIndex{:first}) = 1
getindex(arr::Tensor, axis::LegIndex{AXIS}) where AXIS = Leg(arr, axis)
getindex(arr::Tensor, s::Symbol) = getindex(arr, LegIndex{s}())

################## Bonds #######################
struct Bond{C, L<:Leg{C}}
    leg1::L
    leg2::L
end
