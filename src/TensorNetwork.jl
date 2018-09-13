const Tensor = AbstractArray

abstract type AbstractTN{T} end

"""
Leg(ts::Tensor{T}, axis::Int) where T = Leg{T, size(ts, axis)}(ts, axis)

"""
struct Leg{T, N}
    ts::Array{T}
    axis::Int
end

Leg(ts::Tensor{T}, axis::Int) where T = Leg{T, size(ts, axis)}(ts, axis)

struct Bond{T, N}
    leg1::Leg{T, N}
    leg2::Leg{T, N}
end

length(leg::Leg) = size(leg.ts, leg.axis)
parent(leg::Leg) = leg.ts
