using FunnyTN
using FunnyTN.TensorNetworks
using TensorOperations, LinearAlgebra
using Test

@testset "MPS" begin
    include("MPS.jl")
end

@testset "linalg" begin
    include("linalg.jl")
end
