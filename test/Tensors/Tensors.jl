using FunnyTN
using FunnyTN.Tensors
using TensorOperations, LinearAlgebra
using Test

@testset "linalg" begin
    include("linalg.jl")
end

@testset "interfaces" begin
    include("interfaces.jl")
end
