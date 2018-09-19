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
    "location": "#Manual-1",
    "page": "Home",
    "title": "Manual",
    "category": "section",
    "text": "Pages = [\n    \"funnytn.md\",\n]\nDepth = 1"
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": ""
},

{
    "location": "funnytn/#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "funnytn/#FunnyTN.MPO",
    "page": "Manual",
    "title": "FunnyTN.MPO",
    "category": "type",
    "text": "MPO{T} <: TensorTrain\n\nMatrix Product Operator.\n\nWe use the following convention to number legs:     2     |  1–A–4     |     3\n\nllink -> 1 ulink -> 2 dlink -> 3 rlink -> 4\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN.MPOTensor",
    "page": "Manual",
    "title": "FunnyTN.MPOTensor",
    "category": "type",
    "text": "MPO Tensor\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN.MPSTensor",
    "page": "Manual",
    "title": "FunnyTN.MPSTensor",
    "category": "type",
    "text": "MPS Tensor\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN.vec2mps-Union{Tuple{Array{T,1}}, Tuple{T}} where T",
    "page": "Manual",
    "title": "FunnyTN.vec2mps",
    "category": "method",
    "text": "Parse a normal state into a Matrix produdct state.\n\nstate:     The target state, 1D array. nflavor:     The dimension of a single site, integer. l:     The division point of left and right canonical scanning, integer between 0 and number of site. method:     The method to extract orthonormal matrices.     * :qr  -> get A,B matrices by the method of QR decomposition, faster, rank revealing in a non-straight-forward way.     * :svd  -> get A,B matrices by the method of SVD decomposition, slow, rank revealing. tol:     The tolerence of singular value, float.\n\nReturn an <MPS> instance.\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN.bond",
    "page": "Manual",
    "title": "FunnyTN.bond",
    "category": "function",
    "text": "bond(tt::AbstractTN, l::Int) -> Bond\n\nl-th bound of tensor train.\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN.log2i",
    "page": "Manual",
    "title": "FunnyTN.log2i",
    "category": "function",
    "text": "log2i(x::Integer) -> Integer\n\nReturn log2(x), with integer input only.\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN.tensors",
    "page": "Manual",
    "title": "FunnyTN.tensors",
    "category": "function",
    "text": "tensors(tt::AbstractTN) -> Tensor\n\nGet the list tensors.\n\n\n\n\n\n"
},

{
    "location": "funnytn/#FunnyTN-1",
    "page": "Manual",
    "title": "FunnyTN",
    "category": "section",
    "text": "Modules = [FunnyTN]\nOrder   = [:module, :constant, :type, :macro, :function]"
},

]}
