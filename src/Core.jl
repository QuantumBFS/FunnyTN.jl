const Tensor = AbstractArray
abstract type AbstractTN{T, TT} end
abstract type TensorTrain{T, TT}<:AbstractTN{T, TT} end

"""
    bond(tt::AbstractTN, l::Int) -> Bond

l-th bound of tensor train.
"""
function bond end

"""
    tensors(tt::AbstractTN) -> Tensor

Get the list tensors.
"""
function tensors end

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
