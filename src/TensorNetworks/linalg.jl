⊕(isfirst::Bool, islast::Bool, bc::Symbol, ts::Tensor...) = ⊕(Val{isfirst}, Val{islast}, Val{bc}, ts...)
⊕(::Type{<:Val}, ::Type{<:Val}, ::Type{<:Val}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=(1, N))
⊕(::Type{Val{false}}, ::Type{Val{true}}, ::Type{Val{:open}}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=1)
⊕(::Type{Val{true}}, ::Type{Val{false}}, ::Type{Val{:open}}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=N)
⊕(::Type{Val{true}}, ::Type{Val{true}}, ::Type{Val{:open}}, ts::Tensor{T, N}...) where {T, N} = sum(ts, dims=1)

"""
Summation over <MPS>es.

Args:
    tts (list of <MPS>): instances to be added.
    labels (list of str): the new labels for added state.

Returns:
    <MPS>, the added MPS.
"""
function sum(tts::Vector{<:MPSO{BC, T, N, TT}}) where {T, BC, N, TT}
    length(tts) == 0 && raise(ArgumentError("At least one instances are required."))
    length(tts) == 1 && return tts[0]
    all_equivalent([l_canonical(tt) for tt in tts]) || raise(ArgumentError("canonicality does not match!"))

    tt0 = tts[1]
    l = tt0.l
    hndim = nflavor(tt0)
    nbit = nsite(tt0)
    MLs = [[tensors(mps)...] for mps in tts]

    # get S matrix
    ML = Vector{TT}(undef, nbit)
    if BC == :open && (l == 0 || l==nbit)  # make open open
        S = T[1]
        for (tt, m) in zip(tts, MLs)
            m[1] *= tt.S[]
        end
    else
        S = cat([singular_values(tt) for tt in tts]..., dims=1)
    end
    for (i, mis) in enumerate(zip(MLs...))
        ML[i] = ⊕(i==1, i==nbit, BC, mis...)
    end
    MPS{BC}(ML, l=>S)
end

+(mps1::T, mps2::T) where T<:MPS = sum([mps1, mps2])
rmul!(A::MPS, b::Number) = (rmul!(A.S, b); A)
lmul!(a::Number, B::MPS) = (lmul!(a, B.S); B)
-(A::MPS) = (-1)*A
-(A::MPS, B::MPS) = A + (-1)*B
*(A::MPS, b::Number) = rmul!(copy(A), b)
/(A::MPS, b::Number) = rmul!(copy(A), 1/b)
*(a::Number, B::MPS) = lmul!(a, copy(B))

#################### TensorTrain ###############
function vec(tt::TensorTrain{T}) where T
    B = bondsize(tt, 0)
    res = Matrix{T}(I, B, B)
    for i=1:tt.l
        res = reshape(res ∘ tt[i], :, bondsize(tt, i))
    end
    mulaxis!(res, tt.S, 2)
    for i=tt.l+1:nsite(tt)
        res = reshape(res ∘ tt[i], :, bondsize(tt, i))
    end
    res |> vec
end

##################### MPS ########################
"""
Parse a normal state into a Matrix produdct state.

state:
    The target state, 1D array.
nflavor:
    The dimension of a single site, integer.
l:
    The division point of left and right canonical scanning, integer between 0 and number of site.
method:
    The method to extract orthonormal matrices.
    * :qr  -> get A,B matrices by the method of QR decomposition, faster, rank revealing in a non-straight-forward way.
    * :svd  -> get A,B matrices by the method of SVD decomposition, slow, rank revealing.
tol:
    The tolerence of singular value, float.

Return an <MPS> instance.
"""
function vec2mps(v::Vector{T}; l::Int=0, nflavor::Int=2, method=:SVD, tol=1e-15) where T
    nsite = nqubits(v)
    state = reshape(v, :, 1)
    l >= 0 && l <= nsite || throw(BoundsError("canonical index out of range!"))

    # right mover
    ML = Vector{MPSTensor{T}}(undef, nsite)
    ri = 1
    for i = 1:l
        state = reshape(state, nflavor * ri, :)
        U, state = decompose(Symbol(method, :_R), state)
        ri = size(U, 2)
        ML[i] = reshape(U, :, nflavor, ri)
    end

    # left mover
    ri = 1
    for i in 1:nsite-l
        state = reshape(state, :, nflavor * ri)
        state, V = decompose(Symbol(method, :_L), state)
        ri = size(V, 1)
        ML[nsite-i+1] = reshape(V, ri, nflavor, :)
    end
    S = state |> diag
    mps(ML, l=>S)
end

function decompose(::Type{Val{:QR_R}}, state::Matrix; tol=0)
    U, state = qr(state)
    kpmask = sum(norm, state, dims=2) > tol
    ri = kpmask |> sum
    state = state[kpmask]
    U = U[:, kpmask]
    U, state
end

function decompose(::Type{Val{:QR_L}}, state::Matrix; tol=0)
    state, V = rq(state)
    kpmask = sum(norm, state, dims=1) .> tol
    ri = kpmask.sum()
    state = state[:, kpmask]
    V = V[kpmask]
    state, V
end

function decompose(::Type{Val{:SVD_L}}, state::Matrix; tol=0)
    res = svd(state)
    # remove zeros from v
    kpmask = abs.(res.S) .> tol
    ri = kpmask |> sum
    state = mulaxis!(res.U[:, kpmask], res.S[kpmask], 2)
    V = res.Vt[kpmask, :]
    state, V
end

function decompose(::Type{Val{:SVD_R}}, state::Matrix; tol=0)
    res = svd(state)
    # remove zeros from v
    kpmask = abs.(res.S) .> tol
    ri = kpmask |> sum
    state = mulaxis!(res.Vt[kpmask,:], res.S[kpmask], 1)
    U = res.U[:, kpmask]
    U, state
end

function decompose(s::Symbol, state::Matrix; tol=0)
    decompose(Val{s}, state, tol=tol)
end
