using FunnyTN.TensorNetworks
using FunnyTN.Tensors
using Random
using LinearAlgebra

Random.seed!(2)
const mps0 = rand_mps(ComplexF64, 2, [1,5,8, 16, 64,100, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 200, 100, 80, 40, 20, 6, 1])
mps0 |> normalize!

function compress_fidelity(func)
    mps1 = copy(mps0)
    res = func(mps1, 20)
    println(res)
    res'*mps0
end

#@show compress_fidelity((mps, D)->compress!(mps, D, tol=0, niter=3))
println(naive_compress!(copy(mps0), 20, method=:SVD))
@show compress_fidelity((mps, D)->naive_compress!(mps, D, method=:QR))
mps = mps0 |> copy
using BenchmarkTools
@benchmark naive_compress!($mps, 20)
@show compress_fidelity((mps, D)->canomove!(mps, nsite(mps)-1, D=D))

using FunnyTN.TensorNetworks: decompose, _truncmask

function TensorNetworks.decompose(::Val{:QR}, ::Val{:right}, state::Matrix; D::Int=typemax(Int), tol=0)
    println("@", state|>size)
    F = qr(state)
    S = dropdims(sum(norm, F.R, dims=2), dims=2)
    println(tol,"  ", D)
    kpmask = _truncmask(S, tol, D)
    println("@", state |> size, "@", Matrix(F.Q) |> size, "@", F.R |> size)
    state = F.R[kpmask,:]
    println(kpmask |> sum, "  ", state|>size)
    U = Matrix(F.Q)[:, kpmask]
    U, state
end

function TensorNetworks.decompose(::Val{:QR}, ::Val{:left}, state::Matrix; D::Int=typemax(Int), tol=0)
    state, V = rq(state)
    kpmask = _truncmask(dropdims(sum(norm, state, dims=1), dims=1), tol, D)
    state = state[:, kpmask]
    V = V[kpmask, :]
    state, V
end
