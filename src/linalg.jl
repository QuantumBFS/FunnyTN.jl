glue(t1::Tensor{T1, N1}, t2::Diagonal) where {T1, N1} = Leg(t1, N1) * t2.diag
glue(t1::Tensor{T1, N1}, t2::Tensor) where {T1, N1} = glue(Leg(t1, N1), Leg(t2, 1))
glue(t1::Diagonal, t2::Tensor{T2, N2}) where {T2, N2} = t1.diag * Leg(t2, N2)
glue(s::UniformScaling, t2::Tensor) = s*t2
glue(t2::Tensor, s::UniformScaling) = t2*s
glue(legs::Leg...) = reduce(glue, legs)
function glue(l1::Leg{C1, AT1}, l2::Leg{C2, AT2}) where {C1, C2, T1, T2, N1, N2, AT1<:Tensor{T1, N1}, AT2<:Tensor{T2, N2}}
    ts1 = parent(l1)
    ts2 = parent(l2)
    labels1 = collect(1:N1)
    labels2 = collect(N1+1:N1+N2)
    for (a1, a2) in zip(l1.axes, l2.axes)
        labels2[a2] = labels1[a1]
    end
    tensorcontract(ts1, labels1, ts2, labels2)
end

function Base.vec(mps::MPS)
    res = I
    for i=1:mps.l
        res = reshape(res ∾ mps[i], :, bondsize(mps, i))
    end
    res *= Diagonal(mps.S)
    for i=mps.l+1:nsite(mps)
        res = reshape(res ∾ mps[i], :, bondsize(mps, i))
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
    state = res.U[:, kpmask] * Diagonal(res.S[kpmask])
    V = res.Vt[kpmask, :]
    state, V
end

function decompose(::Type{Val{:SVD_R}}, state::Matrix; tol=0)
    res = svd(state)
    # remove zeros from v
    kpmask = abs.(res.S) .> tol
    ri = kpmask |> sum
    state = Diagonal(res.S[kpmask]) * res.Vt[kpmask,:]
    U = res.U[:, kpmask]
    U, state
end

function decompose(s::Symbol, state::Matrix; tol=0)
    decompose(Val{s}, state, tol=tol)
end
