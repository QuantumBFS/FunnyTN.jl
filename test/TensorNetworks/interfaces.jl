using FunnyTN.Tensors
using FunnyTN.TensorNetworks
using TensorOperations, LinearAlgebra
using Test

@testset ">>, <<" begin
    mps = rand_mps(2, [1,3,4,5,1])
    @test (mps >> 2) |> l_canonical == 2
    @test (mps << 1) |> l_canonical == 1
end
