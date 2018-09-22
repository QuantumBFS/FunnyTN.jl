function mulaxis!(A::AbstractArray, v::AbstractVector, axis::Int)
    for i in 1:size(A, axis)
        rmul!(selectdim(A, axis, i), v[i])
    end
    A
end
mulaxis!(A::AbstractArray, v::Diagonal, axis::Int) = mulaxis!(A, v.diag, axis)

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
function glue(l1::Leg{C1, AT1}, l2::Leg{C2, AT2}) where {C1, C2, T1, T2, N1, N2, AT1<:Tensor{T1, N1}, AT2<:Tensor{T2, N2}}
    ts1 = parent(l1)
    ts2 = parent(l2)
    labels1 = 1:N1
    labels2 = collect(N1+1:N1+N2)
    for (a1, a2) in zip(l1.axes, l2.axes)
        labels2[a2] = labels1[a1]
    end
    tensorcontract(ts1, labels1, ts2, labels2)
end
glue(l::Leg{1}, v::Vector) = glue(l, Leg(v, 1))
glue(v::Vector, l::Leg{1}) = glue(Leg(v, 1), l)
