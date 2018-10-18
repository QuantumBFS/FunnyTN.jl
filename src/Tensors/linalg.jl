function mulaxis!(A::AbstractArray, axis::Int, v::AbstractVector)
    size(A, axis) == length(v) || throw(DimensionMismatch("can not multiple vector of length $(v |>length) on axis of size $(size(A, axis))!"))
    for i in 1:size(A, axis)
        rmul!(selectdim(A, axis, i), v[i])
    end
    A
end
mulaxis!(A::AbstractArray, axis::Int, v::Diagonal) = mulaxis!(A, axis, v.diag)
mulaxis(A::AbstractArray, axis::Int, v) = mulaxis!(copy(A), axis, v)

function chain_tensors(t1::Tensor{T1, N1}, t2::Tensor) where {T1, N1}
    shape1 = size(t1)
    shape2 = size(t2)
    res = reshape(t1, :, size(t1, N1)) * reshape(t2, size(t2, 1), :)
    reshape(res, shape1[1:end-1]..., shape2[2:end]...)
end

function chain_tensors(t1::Diagonal, t2::Tensor)
    shape2 = size(t2)
    reshape(t1*reshape(t2, size(t2, 1), :), size(t1, 1), shape2[2:end]...)
end

function chain_tensors(t2::Tensor{T2, N2}, t1::Diagonal) where {T2, N2}
    shape2 = size(t2)
    reshape(reshape(t2, :, size(t2, N2))*t1, shape2[1:end-1]..., size(t1, 2))
end

chain_tensors(ts::Tensor...) = reduce(chain_tensors, ts)

glue(legs::Leg...) = reduce(glue, legs)
function glue(l1::Leg{AXIS1, AT1}, l2::Leg{AXIS2, AT2}) where {AXIS1, AXIS2, T1, T2, N1, N2, AT1<:Tensor{T1, N1}, AT2<:Tensor{T2, N2}}
    ts1 = parent(l1)
    ts2 = parent(l2)
    labels1 = 1:N1
    labels2 = collect(N1+1:N1+N2)
    labels2[AXIS2] = labels1[AXIS1]
    tensorcontract(ts1, labels1, ts2, labels2)
end
glue(l::Leg, v::Vector) = glue(l, Leg(v, 1))
glue(v::Vector, l::Leg) = glue(Leg(v, 1), l)


"""
    tt_dadd(ts::Tensor{T, N}...) -> Tensor{T, N}

tensor train direct add.
"""
tt_dadd(ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=(1, N))

function rq!(A::AbstractMatrix)
    M, N = size(A)
    rq, tau = LAPACK.gerqf!(A)
    if N<M
        R = triu(rq, N-M)
        Q = LAPACK.orgrq!(rq[M-N+1:M, :], tau)
    else
        R = triu(view(rq, :,N-M+1:N))
        Q = LAPACK.orgrq!(rq, tau)
    end
    R, Q
end
rq(A) = rq!(copy(A))

using Base: has_offset_axes
function LinearAlgebra.triu!(M::AbstractMatrix, k::Integer)
    @assert !has_offset_axes(M)
    m, n = size(M)
    for j in 1:min(n, m + k)
        for i in max(1, j - k + 1):m
            M[i,j] = zero(M[i,j])
        end
    end
    M
end

#=
function _glue(l1::NLeg{C1, AT1}, l2::NLeg{C2, AT2}) where {C1, C2, T1, T2, N1, N2, AT1<:Tensor{T1, N1}, AT2<:Tensor{T2, N2}}
    ts1 = parent(l1)
    ts2 = parent(l2)
    labels1 = 1:N1
    labels2 = collect(N1+1:N1+N2)
    for (a1, a2) in zip(l1 |> axis, l2 |> axis)
        labels2[a2] = labels1[a1]
    end
    tensorcontract(ts1, labels1, ts2, labels2)
end
=#
