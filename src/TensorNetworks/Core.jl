abstract type AbstractTN{T, TT} end
abstract type TensorTrain{T, TT}<:AbstractTN{T, TT} end

function tensors end

"""
    assert_valid(tn)

assert tensor network structure `tn` is valid.
"""
function assert_valid end
