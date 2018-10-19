using FunnyTN.Tensors
using TensorOperations, LinearAlgebra
using Test

@testset "contractions" begin
    include("contractions.jl")
end

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

@testset "mulaxis! and *" begin
    a = randn(4, 7, 5)
    v = randn(4)
    amat = reshape(a, 4,:)
    res = reshape(Diagonal(v) ∘ amat, 4, 7, 5)
    @test size(permutedims(a, [3,2,1]) ∘ a) == (5,7,7,5)
    @test permutedims(a, [3,2,1]) ∘ Diagonal(v) == permutedims(res, [3,2,1])
    @test res ≈ mulaxis!(a |> copy, 1, v)
    @test res ≈ mulaxis!(a |> copy, 1, Diagonal(v))
end

@testset "glue" begin
    a = randn(4, 7, 5)
    v = randn(4)
    amat = reshape(a, 4,:)
    res = reshape(v'*amat, 7, 5)

    @test res ≈ glue(v, Leg(a, 1))
    @test res ≈ glue(Leg(a, 1), v)
end

@testset "rq, triu" begin
    mat=[1 2;
         3 4;
         5 6;
         7 8]
    res=[1 2;
         3 4;
         0 6;
         0 0]
    @test triu!(mat, -1) == res

    a = randn(16, 4)
    R, Q = rq!(copy(a))
    @test R*Q ≈ a

    a = randn(4, 16)
    R, Q = rq!(copy(a))
    @test R*Q ≈ a
end
