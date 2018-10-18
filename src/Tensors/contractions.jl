"""
    tt_dadd(ts::Tensor{T, N}...) -> Tensor{T, N}

tensor train direct add.
"""
tt_dadd(ts::Tensor{T, N}...) where {T, N} = cat(ts..., dims=(1, N))

"""
    bra_ket_prod(A::MPSTensor, B::MPSTensor=A)

bra-ket contraction, returns

    --A*--
      |
    --B---
"""
function bra_ket_prod(A::MPSTensor, B::MPSTensor=A)
    @tensor T[j1,i1,j2,i2] := B[i1,p,i2]*conj(A)[j1,p,j2]
end

"""
    mpo_ket_prod(O::MPOTensor, B::MPSTensor)

mpo-mps contract, returns

      |
    --O*--
      |   
    --B---
"""
function mpo_ket_prod(O::MPSTensor, B::MPOTensor)
    @tensor T[j1,i1,s,j2,i2] := O[j1,s,c,j2] * B[i1,c,i2]
end

"""
    x_bra_ket_prod(DIRECTION, X::Matrix, A::MPSTensor, B::MPSTensor=A)

matrix-bra-ket contraction, returns

DIRECTION == :left
    --A*--⊤
      |   X
    --B---⊥

DIRECTION == :right
    ⊤--A*--
    X  |  
    ⊥--B---
"""
x_bra_ket_prod(direction::Symbol, args...) = x_bra_ket_prod(Val(direction), args...)
function x_bra_ket_prod(::Val{:right}, X::Matrix, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := (X[a, b] * B[b, p, j]) * conj(A[a, p, i])
end
function x_bra_ket_prod(::Val{:right}, ::UniformScaling{Bool}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := B[a, p, j] * conj(A[a, p, i])
end
function x_bra_ket_prod(::Val{:left}, X::Matrix, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := (X[i, j] * B[b, p, j]) * conj(A[a, p, i])
end
function x_bra_ket_prod(::Val{:left}, ::UniformScaling{Bool}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := B[b, p, i] * conj(A[a, p, i])
end

"""
    ket_bra_prod(DIRECTION, T::TMatrix, A::MPSTensor, B::MPSTensor=A)

transfer_tensor-bra-ket contraction, returns

DIRECTION == :left
    --A*--⊤--
      |   T
    --B---⊥--
DIRECTION == :right
    --⊤--A*--
      T  |  
    --⊥--B---
"""
t_bra_ket_prod(direction::Symbol, args...) = t_bra_ket_prod(Val(direction), args...)

function t_bra_ket_prod(::Val{:right}, T::TMatrix, A::MPSTensor, B::MPSTensor=A)
    @tensor T[j0,i0,j2,i2] := (T[j0,i0,j1,i1]*B[i1,p,i2]) * conj(A)[j1,p,j2]
end
function t_bra_ket_prod(::Val{:left}, T::TMatrix, A::MPSTensor, B::MPSTensor=A)
    @tensor T[j2,i2,j1,i1] := (T[j0,i0,j1,i1]*B[i2,p,i0]) * conj(A)[j2,p,j0]
end
