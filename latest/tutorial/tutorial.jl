# # Tutorial
using FunnyTN.TensorNetworks
using LinearAlgebra

# ## Matrix Product State
isnormalized(mps::MPS) = norm(mps) ≈ 1

# ### Example: compressing
# Let's prepair an MPS and two copies of them
mps0 = rand_mps(2, [1,5,8, 16, 64,100, 400, 400, 200, 200, 100, 80, 40, 20, 6, 1], l=1)
mps0 |> normalize!
@assert mps0 |> isnormalized
mps1 = copy(mps0)
mps2 = copy(mps0);

# Now, let's compress `mps1` and `mps2` in two different ways, sweep and direct.
# We will see sweep has higher fidelity than direct compress.
compress!(mps1, 20)
canomove!(mps2, nsite(mps2)-1, D=20)
@show mps1'*mps0
@show mps2'*mps0;
