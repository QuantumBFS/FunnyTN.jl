using FunnyTN.Tensors
using FunnyTN.TensorNetworks
using TensorOperations, LinearAlgebra
using Test

@testset "mpstensor" begin
    ts1 = randn(ComplexF64, 8,2,8)
    ts2 = randn(ComplexF64, 8,2,10)
    l1 = Leg(ts1, 3)
    l2 = Leg(ts2, 1)
    @test l2 |> length == 8
    @tensor D[a,b,d,e] := parent(l1)[a,b,c] * parent(l2)[c,d,e]
    @test glue(Leg(ts1, 3), Leg(ts2, 1)) ≈ D
    @test glue(Leg(ts1, 3), Leg(ts2, 1)) ≈ glue(lastleg(ts1), ts2 |> firstleg)
    @test (ts1 ∘ ts2) ≈ glue(ts1 |> lastleg, ts2 |> firstleg)
end

@testset "constructor" begin
    m = rand_mps(2, [1,8,8,8,8,8,8,8,1])
    @test bondsize(m, 1) == 8
    @test bondsize(m, 0) == bondsize(m, 8) == 1
end
