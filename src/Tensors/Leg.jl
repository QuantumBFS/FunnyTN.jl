################### Legs ###########################
struct Leg{C, AT<:Tensor}
    ts::AT
    axes::NTuple{C, Int}
end
Leg(ts::Tensor{T}, axes::Int...) where T = Leg{length(axes), typeof(ts)}(ts, axes)
lastleg(ts::Tensor{T, N}) where {T, N} = Leg(ts, N)
firstleg(ts::Tensor) = Leg(ts, 1)

length(leg::Leg) = prod(axis->size(leg.ts, axis), leg.axes)
parent(leg::Leg) = leg.ts
Base.show(io::IO, leg::Leg) = print(io, "$(leg.ts |> typeof)$(leg.ts |> size) ⧷ $(leg.axes)")

################## Bonds #######################
struct Bond{C, L<:Leg{C}}
    leg1::L
    leg2::L
end
⧷(t::Tensor, axes::Tuple) = Leg(t, axes...)
⧷(t::Tensor, i::Int) = t ⧷ (i,)
