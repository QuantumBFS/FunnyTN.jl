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

@testset "decompose" begin
    v = randn(2, 4)
    left, right = decompose(:SVD, :left, v)
    @test left*right ≈ v
    @test right*right' ≈ I
    left, right = decompose(:SVD, :right, v)
    @test left*right ≈ v
    @test left'*left ≈ I
end

@testset "mps & vec" begin
    v = randn(ComplexF64, 1<<7)
    @test v |> vec2mps |> vec ≈ v
    @test vec2mps(v, l=2) |> vec ≈ v
end

@testset "scalar arithmatics" begin
    for l = [0, 3, 7]
        mps1 = rand_mps(ComplexF64, 2, [1,2,4,8,8,4,2,1], l=l)
        v1 = mps1 |> vec
        @test mps1*2 |> vec ≈ v1*2
        @test mps1/2 |> vec ≈ v1/2
        @test 2*mps1 |> vec ≈ v1*2
        @test -mps1 |> vec ≈ -v1
    end
end

@testset "mps arithmatics" begin
    for l = [0, 3, 7]
        mps1 = rand_mps(ComplexF64, 2, [1,2,4,8,8,4,2,1], l=l)
        v1 = mps1|>vec
        v1 == mps1 |> vec
        @test isapprox.(mps1 - mps1 |> vec, 0, atol=1e-12) |> all
        mps1 + mps1 |> vec ≈ v1*2
        v2 = sum([mps1, mps1]) |> vec
        @test v2 ≈ v1 *2
    end
end


@testset "canomove" begin
    mps = rand_mps(2, [1,2,4,4,2,1])
    v0 = mps |> vec
    @test_throws ArgumentError canomove!(mps, :left) |> println

    canomove!(mps, :right)
    @test assert_valid(mps)
    @test l_canonical(mps) == 1
    canomove!(mps, 4)
    @test l_canonical(mps) == 5
    canomove!(mps, -4)
    @test l_canonical(mps) == 1
    canomove!(mps, :left)
    @test l_canonical(mps) == 0
    canomove!(mps, 0)
    @test l_canonical(mps) == 0
    v1 = mps |> vec
    @test v0 ≈ v1
end

@testset "ket bra contract" begin
    mps = rand_mps(2,[1,2,3,4,2,1])
    v = mps |> vec
    @test mps' * mps ≈ v'*v
    @test inner_product(Val(:left), mps', mps) ≈ v'*v
end
