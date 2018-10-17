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
    all_equivalent([cloc(tt) for tt in tts]) || raise(ArgumentError("canonicality does not match!"))

    tt0 = tts[1]
    l = tt0.l
    hndim = nflavor(tt0)
    nbit = nsite(tt0)
    MLs = [[tensors(mps)...] for mps in tts]
    ML = [⊕(i==1, i==nbit, BC, mis...) for (i, mis) in enumerate(zip(MLs...))]
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
function vec2mps(v::Vector{T}; l::Int=1, nflavor::Int=2, method=:SVD, tol=1e-15) where T
    nsite = nqubits(v)
    state = reshape(v, :, 1)
    l >= 0 && l <= nsite || throw(BoundsError("canonical index out of range!"))

    # right mover
    ML = Vector{MPSTensor{T}}(undef, nsite)
    ri = 1
    for i = 1:l-1
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
    ML[l] = reshape(state, :, nflavor, ri)
    MPS(ML, l)
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

"""
    canomove!(mps::MPS, direction::Symbol; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD) -> MPS

move canonical center, direction can be :left, :right or Int (Int means moving right for n step).
"""
function canomove!(mps::MPS, direction::Symbol; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD)
    # check and prepair data
    nbit = mps |> nsite
    l_ = mps.l + (direction == :right ? 1 : -1)

    # bounds check
    (l_>nbit || l_<=0) && throw(ArgumentError("Illegal Move! l=$l_"))
    l1, l2 = min(l_, mps.l), max(l_, mps.l)

    nflv = nflavor(mps)
    A, B = mps[l1], mps[l2]

    AB = reshape(A, :, size(A, 3)) * reshape(B, size(B, 1), :)
    A_, S_, B_ = svdtrunc(AB, D=D, tol=tol)
    direction == :right ? mulaxis!(B_, 1, S_) : mulaxis!(A_, 2, S_)
    mps[l1] = reshape(A_, :, nflv, size(A_, 2))
    mps[l2] = reshape(B_, size(B_, 1), nflv, :)
    mps.l = l_
    mps
end

function canomove!(mps::MPS, direction::Integer; tol::Real=1e-15, D::Int=typemax(Int64), method=:SVD)
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

function braket_contract(::Val{:right}, X::Matrix, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := (X[a, b] * B[b, p, j]) * conj(A[a, p, i])
end

function braket_contract(::Val{:right}, ::UniformScaling{Bool}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := B[a, p, j] * conj(A[a, p, i])
end

function braket_contract(::Val{:left}, X::Matrix, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := (X[i, j] * B[b, p, j]) * conj(A[a, p, i])
end

function braket_contract(::Val{:left}, ::UniformScaling{Bool}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := B[b, p, i] * conj(A[a, p, i])
end

braket_contract(direction::Symbol, args...) = braket_contract(Val(direction), args...)

function inner_product(::Val{:right}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    bra = parent(abra)
    C = braket_contract(:right, I, bra[1], ket[1])
    for i = 2:nsite(ket)
        C = braket_contract(:right, C, bra[i], ket[i])
    end
    C[]
end

function inner_product(::Val{:left}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    N = nsite(ket)
    bra = parent(abra)
    C = braket_contract(:left, I, bra[N], ket[N])
    for i = N-1:-1:1
        C = braket_contract(:left, C, bra[i], ket[i])
    end
    C[]
end

function *(bra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    inner_product(Val(:right), bra, ket)
end

function tmatrix_contract(A::MPSTensor, B::MPSTensor=A)
    @tensor T[j1,i1,j2,i2] := B[i1,p,i2]*conj(A)[j1,p,j2]
end

function tmatrix_contract(::Val{:right}, T::TMatrix, A::MPSTensor, B::MPSTensor=A)
    @tensor T[j0,i0,j2,i2] := (T[j0,i0,j1,i1]*B[i1,p,i2]) * conj(A)[j1,p,j2]
end
function tmatrix_contract(::Val{:left}, T::TMatrix, A::MPSTensor, B::MPSTensor=A)
    @tensor T[j2,i2,j1,i1] := (T[j0,i0,j1,i1]*B[i2,p,i0]) * conj(A)[j2,p,j0]
end
tmatrix_contract(direction::Symbol, args...) = tmatrix_contract(Val(direction), args...)

function tmatrix(::Val{:right}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    bra = parent(abra)
    C = tmatrix_contract(bra[1], ket[1])
    for i = 2:nsite(ket)
        C = tmatrix_contract(:right, C, bra[i], ket[i])
    end
    C
end

function tmatrix(::Val{:left}, abra::Adjoint{<:Any, <:MPS{:open}}, ket::MPS{:open})
    bra = parent(abra)
    N = nsite(ket)
    C = tmatrix_contract(bra[N], ket[N])
    for i = N-1:-1:1
        C = tmatrix_contract(:left, C, bra[i], ket[i])
    end
    C
end
