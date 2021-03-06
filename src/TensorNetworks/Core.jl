abstract type AbstractTN{T, TT} end
abstract type TensorTrain{T, TT}<:AbstractTN{T, TT} end

function tensors end

"""
    assert_valid(tn)

assert tensor network structure `tn` is valid.
"""
function assert_valid end

"""
    CanonicalityError <: Exception

Canonicality error.
"""
struct CanonicalityError <: Exception
    msg::String
end

function show(io::IO, e::CanonicalityError)
    print(io, e.msg)
end

const TNOrConjTN{T, TT} = Union{AbstractTN{T, TT}, Adjoint{<:Any, <:AbstractTN{T, TT}}}
show(io::IO, mime::MIME"text/plain", tt::TNOrConjTN) = show(io, tt)
