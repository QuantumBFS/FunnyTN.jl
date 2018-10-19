using FunnyTN.Tensors
using TensorOperations, LinearAlgebra
using Test

@testset "absorb mpo" begin
    t3 = randn(4,4,3)
    mpo = randn(3,2,2,5)
    t = t3[→] |> absorb_mpo(mpo)
    @test t |> size == (8, 8, 5)
    mpst = randn(6,2,3)
    t = mpst[↑] |> absorb_mpo(mpo)
    @test t |> size == (18, 2, 15)
end

@testset "absorb braket" begin
    # Matrix
    t2 = randn(4,8)
    bra = randn(4,2,5)
    ket = randn(8,2,6)
    t = t2[⇉] |> absorb_bra_ket(bra, ket)
    @test t |> size == (5, 6)

    t2 = randn(5,6)
    t = t2[⇇] |> absorb_bra_ket(bra, ket)
    @test t |> size == (4, 8)

    # TMatrix
    t2 = randn(3,7,4,8)
    t = t2[⇉] |> absorb_bra_ket(bra, ket)
    @test t |> size == (3, 7, 5, 6)

    t2 = randn(5,6,1,2)
    t = t2[⇇] |> absorb_bra_ket(bra, ket)
    @test t |> size == (4, 8, 1, 2)

    # I
    bra = randn(4,2,5)
    ket = randn(4,2,6)
    t = I[⇉] |> absorb_bra_ket(bra, ket)
    @test t |> size == (5, 6)

    bra = randn(4,2,5)
    ket = randn(8,2,5)
    t = I[⇇] |> absorb_bra_ket(bra, ket)
    @test t |> size == (4, 8)
end

@testset "absorb ket" begin
    t3 = randn(4,3)
    ket = randn(3,2,5)
    t = t3[→] |> absorb_ket(ket)
    @test t |> size == (8, 5)
end
