const Tensor = AbstractArray
const MPSTensor{T} = AbstractArray{T, 3}
const MPOTensor{T} = AbstractArray{T, 4}
const TMatrix{T} = AbstractArray{T, 4}

"""
    bond(tt::AbstractTN, l::Int) -> Bond

l-th bound of tensor train.
"""
function bond end

"""
    log2i(x::Integer) -> Integer

Return log2(x), with integer input only.
"""
function log2i end

for N in [8, 16, 32, 64, 128]
    T = Symbol(:Int, N)
    UT = Symbol(:UInt, N)
    @eval begin
        log2i(x::$T) = !signbit(x) ? ($(N - 1) - leading_zeros(x)) : throw(ErrorException("nonnegative expected ($x)"))
        log2i(x::$UT) = $(N - 1) - leading_zeros(x)
    end
end
nqubits(m::AbstractArray) = size(m, 1) |> log2i

"""
    all_equivalent(v) -> Bool

all entries in v are all equivalent.
"""
function all_equivalent(v)::Bool
    res = true
    for i in 1:length(v)-1
        @inbounds res &= v[i] == v[i+1]
    end
    res
end

"""
    assert_samesize(tensors, dim::Int)
"""
assert_samesize(tensors, dim::Int) = all_equivalent([size(t, dim) for t in tensors]) || throw(DimensionMismatch("tensors are not same sized in dimension $dim"))
