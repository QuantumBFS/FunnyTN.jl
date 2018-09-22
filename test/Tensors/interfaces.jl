using FunnyTN.Tensors
using TensorOperations, LinearAlgebra
using Test

@testset "interfaces" begin
    ts = randn(6,5,6)
    @test ts ⧷ 3 == Leg(ts, 3)
    @test (ts ⧷ 3 ∾ ts[←]) == glue(Leg(ts, 3), Leg(ts, 1))
end
