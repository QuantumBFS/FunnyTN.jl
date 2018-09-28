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
    bcond(m) == :open
    @test_throws DimensionMismatch MPS{:periodic}([randn(3,2,4), randn(4,2,4)], 0) |> assert_valid
    @test_throws DimensionMismatch MPS([randn(1,2,4), randn(5,2,1)], 0) |> assert_valid
    @test_throws DimensionMismatch MPS([randn(1,2,4), randn(4,1,1)], 0) |> assert_valid
    @test bondsize(m, 1) == 8
    @test bondsize(m, 0) == bondsize(m, 8) == 1
end

@testset "mps array interfaces" begin
    for l = [0, 3, 7]
        mps1 = rand_mps(ComplexF64, 2, [1,2,4,8,8,4,2,1], l=l)
        @test hsize(mps1) == (1<<7,)
        @test mps1 |> vec ≈ [hgetindex(mps1,i) for i=1:1<<7]
    end
end
