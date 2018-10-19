using FunnyTN.Tensors
using FunnyTN.TensorNetworks
using TensorOperations, LinearAlgebra
using Test

@testset ">>, <<" begin
    mps = rand_mps([1,3,4,5,1])
    @test (mps >> 2) |> cloc == 3
    @test (mps << 1) |> cloc == 2
end
