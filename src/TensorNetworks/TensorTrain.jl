"""
    TensorTrain{T, TT}<:AbstractTN{T, TT}

Chained tensors.
"""
abstract type TensorTrain{T, TT}<:AbstractTN{T, TT} end
@forward TensorTrain.tensors getindex, lastindex, setindex!, iterate, length, eltype, eachindex

# List Behavior.
push!(c::TensorTrain, val::Tensor) = (push!(c |> tensors, val); c)
append!(c::TensorTrain, list) = (append!(c |> tensors, list); c)
prepend!(c::TensorTrain, list) = (prepend!(c |> tensors, list); c)
insert!(c::TensorTrain, key, val) = (insert!(c |> tensors, key, val); c)

bond(tt::TensorTrain, l::Int) = Bond(rlink(tt[l]), llink(tt[l+1]))
bondsize(tt::TensorTrain, l::Int) = l==0 ? size(tt[1], 1) : size(tt[l], ndims(tt[l]))
bondsizes(tt::TensorTrain) = [Tuple(size(t, 1) for t in tt)..., size(tt[end], ndims(tt[end]))]
nsite(tt::TensorTrain) = length(tt |> tensors)

##################### MPSO #################################
"""
Tensor Train with homogeneous tensor rank.
"""
abstract type MPSO{BC, T, N, TT<:Tensor{T, N}} <: TensorTrain{T, TT} end

##################### Size Checks ##########################
"""
    assert_chainable(tensors)
"""
function assert_chainable(tensors)
    local tlast = first(tensors)
    for tthis in tensors[2:end]
        tthis |> firstleg |> length == tlast |> lastleg |> length || throw(DimensionMismatch("tensors are not chainable!"))
        tlast = tthis
    end
end

"""
    assert_boundary_match(tensors, bc::Symbol)
"""
assert_boundary_match(tensors, bc::Symbol) = _assert_boundary_match(tensors, Val{bc})
_assert_boundary_match(tensors, ::Type{Val{:free}}) = true
_assert_boundary_match(tensors, ::Type{Val{:open}}) = tensors |> first |> firstleg |> length == 1 && tensors |> last |> lastleg |> length == 1 || throw(DimensionMismatch("invalid tensor input for :open boundary condition."))
_assert_boundary_match(tensors, ::Type{Val{:periodic}}) = first(tensors) |> firstleg |> length == last(tensors) |> lastleg |> length || throw(DimensionMismatch("invalid tensor input for :periodic boundary condition."))
