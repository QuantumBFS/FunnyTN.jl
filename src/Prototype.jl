using LinearAlgebra
using Lazy, BenchmarkTools, Test
import Base: push!, insert!, setindex!, iterate, append!, prepend!, length, eltype, eachindex, getindex, lastindex, size, parent

const Tensor = AbstractArray

abstract type AbstractTN{T, TT} end
abstract type TensorTrain{T, TT}<:AbstractTN{T, TT} end
@forward TensorTrain.data getindex, lastindex, setindex!, iterate, length, eltype, eachindex

# List Behavior.
push!(c::TensorTrain, val::AbstractMatrix) = (push!(c.data, val); c)
push!(c::TensorTrain, val::AbstractMatrix) = (push!(c.data, val); c)
append!(c::TensorTrain, list) = (append!(c.data, list); c)
prepend!(c::TensorTrain, list) = (prepend!(c.data, list); c)
insert!(c::TensorTrain, key, val) = (insert!(c.data, key, val); c)

################### Legs ###########################
struct Leg{T, N}
    ts::Array{T}
    axis::Int
end
Leg(ts::Tensor{T}, axis::Int) where T = Leg{T, size(ts, axis)}(ts, axis)

length(leg::Leg) = size(leg.ts, leg.axis)
parent(leg::Leg) = leg.ts
show(io::IO, leg::Leg) = println(leg.ts |> size, " ⧷ ", leg.axis)

################## Bonds #######################
struct Bond{T, N}
    leg1::Leg{T, N}
    leg2::Leg{T, N}
end
bond(tt::TensorTrain, l::Int) = Bond(rlink(tt[l]), llink(tt[l+1]))

"""glue multiple legs."""
glue(legs::Leg...) = reduce(glue, legs)
#glue(leg1, leg2) = reduce(glue, legs)


################ MPS ##################
struct MPSTensor{T, AT<:DenseArray{T, 3}} <: DenseArray{T, 3}
    data::AT
    MPSTensor(ts::DenseArray{T, 3}) where T = new{T, typeof(ts)}(ts)
end

@forward MPSTensor.data getindex, setindex!, size, eltype
data(mt::MPSTensor) = mt.data
⧷(t::MPSTensor, i::Int) = Leg(t, i)

@testset "mpstensor" begin
    ts1 = MPSTensor(randn(ComplexF64, 8,2,8))
    ts2 = MPSTensor(randn(ComplexF64, 8,2,10))
    l1 = Leg(ts1, 3)
    l2 = Leg(ts2, 1)
    @test l2 |> length == 8
    @test @tensor D[a,b,c,d] := parent(l1)[a,b,c]
end

llink(tt::MPSTensor) = Leg(tt, 1)
rlink(tt::MPSTensor) = Leg(tt, 3)

slink(tt::MPSTensor) = Leg(tt, 2)

struct MPS{T, AT, TT<:MPSTensor{T, AT}} <: TensorTrain{T, TT}
    data::Vector{T}
    n::Int
end
tensors(mps::MPS) = mps.data

nsite(mps::MPS) = mps.n
