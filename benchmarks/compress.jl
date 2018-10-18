using FunnyTN.TensorNetworks
using Random
Random.seed!(2)

mps0 = rand_mps([1,5,8, 16, 64,100, 400, 400, 200, 200, 100, 80, 40, 20, 6, 1])
mps1 = copy(mps)
mps2 = copy(mps)
@show compress!(mps1, 20)'*mps0
@show canomove!(mps2, nsite(mps2)-1, 20)'*mps0
