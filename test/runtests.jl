using FunnyTN
using TensorOperations, LinearAlgebra
using BenchmarkTools, Test

@testset "mpstensor" begin
    ts1 = randn(ComplexF64, 8,2,8)
    ts2 = randn(ComplexF64, 8,2,10)
    l1 = Leg(ts1, 3)
    l2 = Leg(ts2, 1)
    @test l2 |> length == 8
    @tensor D[a,b,d,e] := parent(l1)[a,b,c] * parent(l2)[c,d,e]
    @test (ts1 ⧷ 3 ∾ ts2 ⧷ 1) ≈ D
    @test (ts1 ⧷ 3 ∾ ts2 ⧷ 1) ≈ (ts1[→] ∾ ts2[←])
    @test (ts1 ∾ ts2) ≈ (ts1[→] ∾ ts2[←])
end

@testset "decompose" begin
    v = randn(2, 4)
    left, right = decompose(:SVD_L, v)
    @test left*right ≈ v
    @test right*right' ≈ I
    left, right = decompose(:SVD_R, v)
    @test left*right ≈ v
    @test left'*left ≈ I
end

@testset "mps" begin
    v = randn(ComplexF64, 1<<7)
    @test v |> vec2mps |> statevec ≈ v
end
