⊕(isfirst::Bool, islast::Bool, bc::Symbol, ts::Tensor...) = ⊕(Val(isfirst), Val(islast), Val(bc), ts...)
⊕(::Val, ::Val, ::Val, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=(1, N))
⊕(::Val{false}, ::Val{true}, ::Val{:open}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=1)
⊕(::Val{true}, ::Val{false}, ::Val{:open}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=N)
⊕(::Val{true}, ::Val{true}, ::Val{:open}, ts::Tensor{T, N}...) where {T, N} = sum(ts, dims=1)

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
    mulaxis!(res, 2, tt.S)
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
        U, state = decompose(method, :right, state)
        ri = size(U, 2)
        ML[i] = reshape(U, :, nflavor, ri)
    end

    # left mover
    ri = 1
    for i in 1:nsite-l
        state = reshape(state, :, nflavor * ri)
        state, V = decompose(method, :left, state)
        ri = size(V, 1)
        ML[nsite-i+1] = reshape(V, ri, nflavor, :)
    end
    S = state |> diag
    MPS(ML, l=>S)
end

function decompose(::Val{:QR}, ::Val{:right}, state::Matrix; D::Int=typemax(Int), tol=0)
    U, state = qr(state)
    kpmask = _truncmask(sum(norm, state, dims=1), tol, D)
    state = state[kpmask]
    U = U[:, kpmask]
    U, state
end

function decompose(::Val{:QR}, ::Val{:left}, state::Matrix; D::Int=typemax(Int), tol=0)
    state, V = rq(state)
    kpmask = _truncmask(sum(norm, state, dims=1), tol, D)
    state = state[:, kpmask]
    V = V[kpmask]
    state, V
end

function _truncmask(S::Vector, tol::Real, D::Int)
    D = min(D, length(res.S))
    mval = max(sort(S, rev=true)[D], tol)
    S .>= mval
end

"""
    svdtrunc(state::Matrix; tol=0)

SVD decomposition and truncate.
"""
function svdtrunc(state::Matrix; D::Int=typemax(Int), tol=0)
    res = svd(state)
    # remove zeros from v
    D = min(D, length(res.S))
    nkeep = D
    for i=1:D
        if res.S[i] < tol
            nkeep = i-1
            break
        end
    end
    res.U[:, 1:nkeep], res.S[1:nkeep], res.Vt[1:nkeep, :]
end

function decompose(::Val{:SVD}, ::Val{:left}, state::Matrix; D::Int=typemax(Int), tol=0)
    U, S, V = svdtrunc(state, tol=tol)
    mulaxis!(U, 2, S), V
end

function decompose(::Val{:SVD}, ::Val{:right}, state::Matrix; D::Int=typemax(Int), tol=0)
    U, S, V = svdtrunc(state, tol=tol)
    U, mulaxis!(V, 1, S)
end

function decompose(method::Symbol, direction::Symbol, state::Matrix; D::Int=typemax(Int), tol=0)
    decompose(Val(method), Val(direction), state, tol=tol)
end

function canomove!(mps::MPS{:open}, ::Val{:_rightmost})
    S = sqrt(sum(mulaxis!(mps[end], 1, mps.S).^2))
    mps.S = [S]
    mps[end] ./= S
    mps.l += 1
    mps
end
function canomove!(mps::MPS{:open}, ::Val{:_leftmost})
    S = sqrt(sum(mulaxis!(mps[1], 3, mps.S).^2))
    mps.S = [S]
    mps[1] ./= S
    mps.l -= 1
    mps
end

"""
    canomove!(mps::MPS, direction::Symbol; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD) -> MPS

move canonical center, direction can be :left, :right or Int (Int means moving right for n step).
"""
function canomove!(mps::MPS, direction::Symbol; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD)
    # check and prepair data
    nbit = mps |> nsite
    l_ = mps.l + (direction == :right ? 1 : -1)

    # bounds check
    l_ == nbit && return canomove!(mps, Val(:_rightmost))
    l_ == 0 && return canomove!(mps, Val(:_leftmost))
    (l_>nbit || l_<0) && throw(ArgumentError("Illegal Move!"))

    nflv = nflavor(mps)
    A, B = mps[l_], mps[l_+1]
    direction == :right ? mulaxis!(A, 1, mps.S) : mulaxis!(B, 3, mps.S)

    AB = reshape(A, :, size(A, 3)) * reshape(B, size(B, 1), :)
    A_, S_, B_ = svdtrunc(AB, D=D, tol=tol)
    mps[l_] = reshape(A_, :, nflv, size(A_, 2))
    mps[l_+1] = reshape(B_, size(B_, 1), nflv, :)
    mps.S = S_
    mps.l = l_
    mps
end

function canomove!(mps::MPS, direction::Integer; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD)
    direction == 0 && return mps
    for i = 1:abs(direction)
        if direction > 0
            canomove!(mps, :right, tol=tol, D=D, method=method)
        else
            canomove!(mps, :left, tol=tol, D=D, method=method)
        end
    end
    mps
end

"""
    compress(mps::MPS, D::Int; tol::Real=1e-15, niter::Int=3, method=:SVD) -> MPS

Compress an mps.
"""
function compress!(mps::MPS, D::Int; tol::Real=1e-15, niter::Int=3, method=:SVD)
    nbit, l = nsite(mps), mps.l
    M = maximum(bondsizes(mps))
    dM = max(M - D, 0)
    for i in 1:niter
        m1 = D + (dM * ((niter - i - 0.5) / niter)) |> round |> Int
        m2 = D + (dM * ((niter - i - 1.0) / niter)) |> round |> Int
        canomove!(mps, nbit - l, tol=tol, D=m1, method=method)
        canomove!(mps, l-nbit, tol=tol, D=m2, method=method)
        canomove!(mps, -l, tol=tol, D=m1, method=method)
        canomove!(mps, l, tol=tol, D=m2, method=method)
    end
    mps
end

"""
    recanonicalize(mps::MPS; move_right_first::Bool=true, tol::Real=1e-15, D::Int=1000) -> MPS

Trun this MPS into canonical form.
"""
function recanonicalize!(mps::MPS; move_right_first::Bool=true, tol::Real=1e-15, D::Int=1000)
    nbit, l = nsite(mps), mps.l
    if move_right_first
        canomove!(mps, nbit - l, tol=tol, D=D)
        canomove!(mps, -nbit, tol=tol, D=D)
        canomove!(mps, l, tol=tol, D=D)
    else
        canomove!(mps, -l, tol=tol, D=D)
        canomove!(mps, nbit, tol=tol, D=D)
        canomove!(mps, l-nbit, tol=tol, D=D)
    end
    mps
end
