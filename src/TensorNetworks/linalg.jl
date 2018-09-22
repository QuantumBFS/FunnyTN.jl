function vec(mps::MPS{T}) where T
    B = bondsize(mps, 0)
    res = Matrix{T}(I, B, B)
    for i=1:mps.l
        res = reshape(res ∘ mps[i], :, bondsize(mps, i))
    end
    mulaxis!(res, mps.S, 2)
    for i=mps.l+1:nsite(mps)
        res = reshape(res ∘ mps[i], :, bondsize(mps, i))
    end
    res |> vec
end

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
