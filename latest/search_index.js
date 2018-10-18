var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "CurrentModule = FunnyTN"
},

{
    "location": "#FunnyTN-1",
    "page": "Home",
    "title": "FunnyTN",
    "category": "section",
    "text": "Funny tensor network (FunnyTN) is the pictograph for tensor networks."
},

{
    "location": "#Tutorial-1",
    "page": "Home",
    "title": "Tutorial",
    "category": "section",
    "text": "Pages = [\n    \"tutorial/tutorial.md\",\n]\nDepth = 1"
},

{
    "location": "#Manual-1",
    "page": "Home",
    "title": "Manual",
    "category": "section",
    "text": "Pages = [\n    \"man/funnytn.md\",\n    \"man/tensors.md\",\n    \"man/tensornetworks.md\",\n]\nDepth = 1"
},

{
    "location": "tutorial/tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": "EditURL = \"https://github.com/QuantumBFS/FunnyTN.jl/blob/master/../../../../build/QuantumBFS/FunnyTN.jl/docs/src/tutorial/tutorial.jl\""
},

{
    "location": "tutorial/tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": "using FunnyTN.TensorNetworks\nusing LinearAlgebra"
},

{
    "location": "tutorial/tutorial/#Matrix-Product-State-1",
    "page": "Tutorial",
    "title": "Matrix Product State",
    "category": "section",
    "text": "isnormalized(mps::MPS) = norm(mps) ≈ 1"
},

{
    "location": "tutorial/tutorial/#Example:-compressing-1",
    "page": "Tutorial",
    "title": "Example: compressing",
    "category": "section",
    "text": "Let\'s prepair an MPS and two copies of themmps0 = rand_mps(2, [1,5,8, 16, 64,100, 400, 400, 200, 200, 100, 80, 40, 20, 6, 1], l=1)\nmps0 |> normalize!\n@assert mps0 |> isnormalized\nmps1 = copy(mps0)\nmps2 = copy(mps0);Now, let\'s compress mps1 and mps2 in two different ways, sweep and direct. We will see sweep has higher fidelity than direct compress.compress!(mps1, 20)\ncanomove!(mps2, nsite(mps2)-1, D=20)\n@show mps1\'*mps0\n@show mps2\'*mps0;This page was generated using Literate.jl."
},

{
    "location": "man/funnytn/#",
    "page": "FunnyTN",
    "title": "FunnyTN",
    "category": "page",
    "text": ""
},

{
    "location": "man/funnytn/#FunnyTN.FunnyTN",
    "page": "FunnyTN",
    "title": "FunnyTN.FunnyTN",
    "category": "module",
    "text": "Pictograph for Tensor Networks is funny!.\n\n\n\n\n\n"
},

{
    "location": "man/funnytn/#FunnyTN-1",
    "page": "FunnyTN",
    "title": "FunnyTN",
    "category": "section",
    "text": "Modules = [FunnyTN]\nOrder   = [:module, :constant, :type, :macro, :function]"
},

{
    "location": "man/tensors/#",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors",
    "category": "page",
    "text": ""
},

{
    "location": "man/tensors/#FunnyTN.Tensors.absorb_bra_ket-Tuple{Symbol,Any,AbstractArray{T,3} where T,AbstractArray{T,3} where T}",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.absorb_bra_ket",
    "category": "method",
    "text": "absorb_bra_ket(DIRECTION, {T::Tensor}, A::MPSTensor, B::MPSTensor=A) -> Tensor\nabsorb_bra_ket(DIRECTION, I, A::MPSTensor, B::MPSTensor=A) -> Tensor\n\ntensor ← bra, ket -> tensor, T can be lazy. Returns\n\nT is rank 2, DIRECTION == :left     –A*–⊤       |   T     –B–-⊥\n\nT is rank2, DIRECTION == :right     ⊤–A*–     T  |     ⊥–B–-\n\nT is rank 4, DIRECTION == :left     –A*–⊤–       |   T     –B–-⊥–\n\nT is rank 4, DIRECTION == :right     –⊤–A*–       T  |     –⊥–B–-\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.absorb_mpo-Tuple{AbstractArray{T,4} where T}",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.absorb_mpo",
    "category": "method",
    "text": "absorb_mpo({B::Tensor}, O::MPOTensor) -> Tensor\n\nmps ← mpo -> mps, returns\n\nB is rank 3            |         /–O*–∖      ==    |    ==         ∖–B–-/\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.all_equivalent-Tuple{Any}",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.all_equivalent",
    "category": "method",
    "text": "all_equivalent(v) -> Bool\n\nall entries in v are all equivalent.\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.assert_samesize-Tuple{Any,Int64}",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.assert_samesize",
    "category": "method",
    "text": "assert_samesize(tensors, dim::Int)\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.bra_ket_prod",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.bra_ket_prod",
    "category": "function",
    "text": "bra_ket_prod(A::MPSTensor, B::MPSTensor=A)\n\nbra-ket contraction, returns\n\n--A*--\n  |\n--B---\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.log2i",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.log2i",
    "category": "function",
    "text": "log2i(x::Integer) -> Integer\n\nReturn log2(x), with integer input only.\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.mpo_ket_prod-Tuple{AbstractArray{T,4} where T,AbstractArray{T,3} where T}",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.mpo_ket_prod",
    "category": "method",
    "text": "mpo_ket_prod(O::MPOTensor, B::MPSTensor)\n\nmpo-mps contract, returns\n\n  |\n--O*--\n  |\n--B---\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.tt_dadd-Union{Tuple{Vararg{AbstractArray{T,N},N1} where N1}, Tuple{N}, Tuple{T}} where N where T",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.tt_dadd",
    "category": "method",
    "text": "tt_dadd(ts::Tensor{T, N}...) -> Tensor{T, N}\n\ntensor train direct add.\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.absorb",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.absorb",
    "category": "function",
    "text": "Tensor Absorption (holomophic transform of tensors)\n\nabsorb_mpo\nabsorbbraket\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors.bond",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors.bond",
    "category": "function",
    "text": "bond(tt::AbstractTN, l::Int) -> Bond\n\nl-th bound of tensor train.\n\n\n\n\n\n"
},

{
    "location": "man/tensors/#FunnyTN.Tensors-1",
    "page": "FunnyTN.Tensors",
    "title": "FunnyTN.Tensors",
    "category": "section",
    "text": "Modules = [FunnyTN.Tensors]\nOrder   = [:module, :constant, :type, :macro, :function]"
},

{
    "location": "man/tensornetworks/#",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks",
    "category": "page",
    "text": ""
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.CanonicalityError",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.CanonicalityError",
    "category": "type",
    "text": "CanonicalityError <: Exception\n\nCanonicality error.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.MPO",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.MPO",
    "category": "type",
    "text": "MPO{T} <: TensorTrain\n\nMatrix Product Operator.\n\nWe use the following convention to number legs:     2     |  1–A–4     |     3\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.MPS",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.MPS",
    "category": "type",
    "text": "MPS{T, BC, TT<:MPSTensor{T}} <: TensorTrain{T, TT}\nMPS{BC}(tensors::Vector{TT}, l::Int) -> MPS\n\nmatrix product state, BC is the boundary condition, l is the canonical center (0 if not needed).\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.MPSO",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.MPSO",
    "category": "type",
    "text": "Tensor Train with homogeneous tensor rank.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.TensorTrain",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.TensorTrain",
    "category": "type",
    "text": "TensorTrain{T, TT}<:AbstractTN{T, TT}\n\nChained tensors.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.assert_boundary_match-Tuple{Any,Symbol}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.assert_boundary_match",
    "category": "method",
    "text": "assert_boundary_match(tensors, bc::Symbol)\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.assert_chainable-Tuple{Any}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.assert_chainable",
    "category": "method",
    "text": "assert_chainable(tensors)\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.assert_valid",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.assert_valid",
    "category": "function",
    "text": "assert_valid(tn)\n\nassert tensor network structure tn is valid.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.canomove!-Tuple{FunnyTN.TensorNetworks.MPS,Symbol}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.canomove!",
    "category": "method",
    "text": "canomove!(mps::MPS, direction::Symbol; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD) -> MPS\n\nmove canonical center, direction can be :left, :right or Int (Int means moving right for n step).\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.compress!-Tuple{FunnyTN.TensorNetworks.MPS,Int64}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.compress!",
    "category": "method",
    "text": "compress!(mps::MPS, D::Int; tol::Real=1e-15, niter::Int=3, method=:SVD) -> MPS\n\nCompress an mps.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.hgetindex-Tuple{FunnyTN.TensorNetworks.MPS,Tuple}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.hgetindex",
    "category": "method",
    "text": "hgetindex(mps::MPS, inds) -> Number\n\nReturns the amplitude of specific configuration in hilbert space, ind can be either integer or tuple.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.hsize-Tuple{FunnyTN.TensorNetworks.MPS}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.hsize",
    "category": "method",
    "text": "hsize(mps::MPS, [i::Int])\n\nsize of hilbert space.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.naive_compress!-Tuple{FunnyTN.TensorNetworks.MPS,Int64}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.naive_compress!",
    "category": "method",
    "text": "naive_compress!(mps::MPS, D::Int; tol::Real=1e-15, method=:SVD) -> MPS\n\nCompress an mps in a naive way.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.rand_mps-Union{Tuple{T}, Tuple{Type{T},Int64,Array{Int64,1}}} where T",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.rand_mps",
    "category": "method",
    "text": "rand_mps([::Type], nflavor::Int, bond_dims::Vector{Int}) -> MPS\n\nRandom matrix product state.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.recanonicalize!-Tuple{FunnyTN.TensorNetworks.MPS}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.recanonicalize!",
    "category": "method",
    "text": "recanonicalize(mps::MPS; move_right_first::Bool=true, tol::Real=1e-15, D::Int=1000) -> MPS\n\nTrun this MPS into canonical form.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.svdtrunc-Tuple{Array{T,2} where T}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.svdtrunc",
    "category": "method",
    "text": "svdtrunc(state::Matrix; tol=0)\n\nSVD decomposition and truncate.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.vec2mps-Union{Tuple{Array{T,1}}, Tuple{T}} where T",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.vec2mps",
    "category": "method",
    "text": "Parse a normal state into a Matrix produdct state.\n\nstate:     The target state, 1D array. nflavor:     The dimension of a single site, integer. l:     The division point of left and right canonical scanning, integer between 0 and number of site. method:     The method to extract orthonormal matrices.     * :qr  -> get A,B matrices by the method of QR decomposition, faster, rank revealing in a non-straight-forward way.     * :svd  -> get A,B matrices by the method of SVD decomposition, slow, rank revealing. tol:     The tolerence of singular value, float.\n\nReturn an <MPS> instance.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#Base.getindex-Tuple{AbstractArray{T,4} where T,FunnyTN.Tensors.LegIndex{:up}}",
    "page": "FunnyTN.TensorNetworks",
    "title": "Base.getindex",
    "category": "method",
    "text": "MPO Tensor\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#Base.sum-Union{Tuple{Array{#s14,1} where #s14<:MPSO{BC,T,N,TT}}, Tuple{TT}, Tuple{N}, Tuple{BC}, Tuple{T}} where TT where N where BC where T",
    "page": "FunnyTN.TensorNetworks",
    "title": "Base.sum",
    "category": "method",
    "text": "Summation over <MPS>es.\n\nArgs:     tts (list of <MPS>): instances to be added.     labels (list of str): the new labels for added state.\n\nReturns:     <MPS>, the added MPS.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks._tt_dadd_bc-Union{Tuple{N}, Tuple{T}, Tuple{Bool,Bool,Symbol,Vararg{AbstractArray{T,N},N1} where N1}} where N where T",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks._tt_dadd_bc",
    "category": "method",
    "text": "in addition of tensor trains, we need to take boundary condition into consideration.\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks.mapaxis-Tuple{AbstractArray{T,3} where T,FunnyTN.Tensors.LegIndex{:up}}",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks.mapaxis",
    "category": "method",
    "text": "MPS Tensor\n\n\n\n\n\n"
},

{
    "location": "man/tensornetworks/#FunnyTN.TensorNetworks-1",
    "page": "FunnyTN.TensorNetworks",
    "title": "FunnyTN.TensorNetworks",
    "category": "section",
    "text": "Modules = [FunnyTN.TensorNetworks]\nOrder   = [:module, :constant, :type, :macro, :function]"
},

]}
