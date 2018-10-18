"""in addition of tensor trains, we need to take boundary condition into consideration."""
_tt_dadd_bc(isfirst::Bool, islast::Bool, bc::Symbol, ts::Tensor{T, N}...) where {T, N} = _tt_dadd_bc(Val(isfirst), Val(islast), Val(bc), ts...)
_tt_dadd_bc(::Val, ::Val, ::Val, ts::Tensor{T, N}...) where {T, N} = tt_dadd(ts...)
_tt_dadd_bc(::Val{false}, ::Val{true}, ::Val{:open}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=1)
_tt_dadd_bc(::Val{true}, ::Val{false}, ::Val{:open}, ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=N)
_tt_dadd_bc(::Val{true}, ::Val{true}, ::Val{:open}, ts::Tensor{T, N}...) where {T, N} = sum(ts, dims=1)

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
    all_equivalent([cloc(tt) for tt in tts]) || raise(ArgumentError("canonicality does not match!"))

    tt0 = tts[1]
    l = tt0.l
    hndim = nflavor(tt0)
    nbit = nsite(tt0)
    MLs = [[tensors(mps)...] for mps in tts]
    ML = [_tt_dadd_bc(i==1, i==nbit, BC, mis...) for (i, mis) in enumerate(zip(MLs...))]
    MPS{BC}(ML, l)
end

+(mps1::T, mps2::T) where T<:MPS = sum([mps1, mps2])
rmul!(A::MPS, b::Number) = (rmul!(A |> ccenter, b); A)
lmul!(a::Number, B::MPS) = (lmul!(a, B |> ccenter); B)
-(A::MPS) = (-1)*A
-(A::MPS, B::MPS) = A + (-1)*B
*(A::MPS, b::Number) = (A_ = copy(A); A_[A.l]*=b; A_)
/(A::MPS, b::Number) = A*(1/b)
*(a::Number, B::MPS) = B*a

norm(mps::MPS) = sqrt(mps'*mps)
normalize!(mps::MPS) = rmul!(mps, 1/norm(mps))

#################### MPS*MPO ###################
*(A::MPO, B::MPS{:open}) = MPS([b |> absorb_mpo(a) for (a, b) in zip(A, B)])

#################### TensorTrain ###############
function vec(tt::TensorTrain{T}) where T
    B = bondsize(tt, 0)
    res = Matrix{T}(I, B, B)
    for i=1:nsite(tt)
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
function vec2mps(v::Vector{T}; l::Int=1, nflavor::Int=2, method=:SVD, D::Int=typemax(Int), tol=1e-15) where T
    nsite = nqubits(v)
    state = reshape(v, :, 1)
    l >= 0 && l <= nsite || throw(BoundsError("canonical index out of range!"))

    # right mover
    ML = Vector{MPSTensor{T}}(undef, nsite)
    ri = 1
    for i = 1:l-1
        state = reshape(state, nflavor * ri, :)
        U, state = decompose(method, :right, state, D=D, tol=tol)
        ri = size(U, 2)
        ML[i] = reshape(U, :, nflavor, ri)
    end

    # left mover
    ri = 1
    for i in 1:nsite-l
        state = reshape(state, :, nflavor * ri)
        state, V = decompose(method, :left, state, D=D, tol=tol)
        ri = size(V, 1)
        ML[nsite-i+1] = reshape(V, ri, nflavor, :)
    end
    ML[l] = reshape(state, :, nflavor, ri)
    MPS(ML, l)
end


function decompose(::Val{:QR}, ::Val{:right}, state::Matrix; D::Int=typemax(Int), tol=0)
    U, state = qr(state)
    S = dropdims(mapslices(norm, state, dims=2), dims=2)
    kpmask = _truncmask(S, tol, D)
    state = state[kpmask,:]
    U_ = Matrix(U)[:, kpmask]
    U_, state
end

function decompose(::Val{:QR}, ::Val{:left}, state::Matrix; D::Int=typemax(Int), tol=0)
    state, V = rq(state)
    S = dropdims(mapslices(norm, state, dims=1), dims=1)
    kpmask = _truncmask(S, tol, D)
    state = state[:, kpmask]
    V = V[kpmask, :]
    state, V
end

function _truncmask(S::Vector, tol::Real, D::Int)
    D = min(D, length(S))
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
    U, S, V = svdtrunc(state, D=D, tol=tol)
    mulaxis!(U, 2, S), V
end

function decompose(::Val{:SVD}, ::Val{:right}, state::Matrix; D::Int=typemax(Int), tol=0)
    U, S, V = svdtrunc(state, D=D, tol=tol)
    U, mulaxis!(V, 1, S)
end

function decompose(method::Symbol, direction::Symbol, state::Matrix; D::Int=typemax(Int), tol=0)
    decompose(Val(method), Val(direction), state, D=D, tol=tol)
end

"""
    canomove!(mps::MPS, direction::Symbol; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD) -> MPS

move canonical center, direction can be :left, :right or Int (Int means moving right for n step).
"""
function canomove!(mps::MPS, direction::Symbol; tol::Real=0, D::Int=typemax(Int64), method=:SVD)
    # check and prepair data
    nbit = mps |> nsite
    l_ = mps.l + (direction == :right ? 1 : -1)

    # bounds check
    (l_>nbit || l_<=0) && throw(ArgumentError("Illegal Move! l=$l_"))
    l1, l2 = min(l_, mps.l), max(l_, mps.l)

    nflv = nflavor(mps)
    A, B = mps[l1], mps[l2]

    AB = reshape(A, :, size(A, 3)) * reshape(B, size(B, 1), :)
    A_, B_ = decompose(method, direction, AB, D=D, tol=tol)
    #A_, S_, B_ = svdtrunc(AB, D=D, tol=tol)
    #direction == :right ? mulaxis!(B_, 1, S_) : mulaxis!(A_, 2, S_)
    mps[l1] = reshape(A_, :, nflv, size(A_, 2))
    mps[l2] = reshape(B_, size(B_, 1), nflv, :)
    mps.l = l_
    mps
end

function canomove!(mps::MPS, direction::Integer; tol::Real=0, D::Int=typemax(Int64), method=:SVD)
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
    compress!(mps::MPS, D::Int; tol::Real=1e-15, niter::Int=3, method=:SVD) -> MPS

Compress an mps.
"""
function compress!(mps::MPS, D::Int; tol::Real=1e-15, niter::Int=3, method=:SVD)
    nbit, l = nsite(mps), mps.l
    M = maximum(bondsizes(mps))
    dM = max(M - D, 0)
    for i in 1:niter
        m1 = D + (dM * ((niter - i + 0.5) / niter)) |> round |> Int
        m2 = D + (dM * ((niter - i) / niter)) |> round |> Int
        canomove!(mps, nbit - l, tol=tol, D=m1, method=method)
        canomove!(mps, l-nbit, tol=tol, D=m2, method=method)
        canomove!(mps, -l+1, tol=tol, D=m1, method=method)
        canomove!(mps, l-1, tol=tol, D=m2, method=method)
    end
    mps
end

"""
    naive_compress!(mps::MPS, D::Int; tol::Real=1e-15, method=:SVD) -> MPS

Compress an mps in a naive way.
"""
function naive_compress!(mps::MPS, D::Int; tol::Real=1e-15, method=:SVD)
    nbit = nsite(mps)
    mps.l = 1
    canomove!(mps, nbit-1, tol=0, D=mps|>bondsizes|>maximum, method=:QR)
    canomove!(mps, -nbit+1, tol=tol, D=D, method=method)
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

function *(bra::MPS, ket::MPS)
    TM = bra[1][↓] ∾ ket[1][↑]
    for i = 1:nsite(bra)
        TM[:topright] * bra[i][←]
        TM[:bottomright] * ket[i][←]
    end
    TM
end

#=
function adjoint!(mps::MPS)
    conj!.(mps |> tensors)
    mps
end
=#

function inner_product(::Val{:right}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    bra = parent(abra)
    C = absorb_bra_ket(:right, I, bra[1], ket[1])
    for i = 2:nsite(ket)
        C = absorb_bra_ket(:right, C, bra[i], ket[i])
    end
    C[]
end

function inner_product(::Val{:left}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    N = nsite(ket)
    bra = parent(abra)
    C = absorb_bra_ket(:left, I, bra[N], ket[N])
    for i = N-1:-1:1
        C = absorb_bra_ket(:left, C, bra[i], ket[i])
    end
    C[]
end

function *(bra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    inner_product(Val(:right), bra, ket)
end

function tmatrix(::Val{:right}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    bra = parent(abra)
    C = bra_ket_prod(bra[1], ket[1])
    for i = 2:nsite(ket)
        C = absorb_bra_ket(:right, C, bra[i], ket[i])
    end
    C
end

function tmatrix(::Val{:left}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    bra = parent(abra)
    N = nsite(ket)
    C = bra_ket_prod(bra[N], ket[N])
    for i = N-1:-1:1
        C = absorb_bra_ket(:left, C, bra[i], ket[i])
    end
    C
end
