using FunnyTN
using TensorOperations, LinearAlgebra
using Test

@testset "Tensors" begin
    include("Tensors/Tensors.jl")
end

@testset "TensorNetwors" begin
    include("TensorNetworks/TensorNetworks.jl")
end
