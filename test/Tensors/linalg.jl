using FunnyTN.Tensors
using TensorOperations, LinearAlgebra
using Test

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
