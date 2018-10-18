"""
Tensor Absorption (holomophic transform of tensors)

* absorb_mpo
* absorb_bra_ket
"""
function absorb end

"""
    absorb_mpo({B::Tensor}, O::MPOTensor) -> Tensor

mps ← mpo -> mps, returns

B is rank 3
           |
        /--O*--∖
     ==    |    ==
        ∖--B---/
"""
absorb_mpo(O::MPOTensor) = B -> absorb_mpo(B, O)
function absorb_mpo(B::MPSTensor, O::MPOTensor)
    ts = mpo_ket_prod(O, B)
    reshape(ts, size(ts, 1)*size(ts, 2), size(ts, 3), size(ts, 4)*size(ts, 5))
end

"""
    absorb_bra_ket(DIRECTION, {T::Tensor}, A::MPSTensor, B::MPSTensor=A) -> Tensor
    absorb_bra_ket(DIRECTION, I, A::MPSTensor, B::MPSTensor=A) -> Tensor

tensor ← bra, ket -> tensor, T can be lazy. Returns

T is rank 2, DIRECTION == :left
    --A*--⊤
      |   T
    --B---⊥

T is rank2, DIRECTION == :right
    ⊤--A*--
    T  |
    ⊥--B---

T is rank 4, DIRECTION == :left
    --A*--⊤--
      |   T
    --B---⊥--

T is rank 4, DIRECTION == :right
    --⊤--A*--
      T  |
    --⊥--B---
"""
absorb_bra_ket(direction::Symbol, T, A::MPSTensor, B::MPSTensor) = absorb_bra_ket(Val(direction), T, A, B)
absorb_bra_ket(direction::Symbol, A::MPSTensor, B::MPSTensor) = T -> absorb_bra_ket(Val(direction), T, A, B)

function absorb_bra_ket(::Val{:right}, T::Tensor, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := (T[a, b] * B[b, p, j]) * conj(A[a, p, i])
end
function absorb_bra_ket(::Val{:right}, ::UniformScaling{Bool}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := B[a, p, j] * conj(A[a, p, i])
end
function absorb_bra_ket(::Val{:left}, T::Tensor, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := (T[i, j] * B[b, p, j]) * conj(A[a, p, i])
end
function absorb_bra_ket(::Val{:left}, ::UniformScaling{Bool}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := B[b, p, i] * conj(A[a, p, i])
end
function absorb_bra_ket(::Val{:right}, T::TMatrix, A::MPSTensor, B::MPSTensor=A)
    @tensor T[j0,i0,j2,i2] := (T[j0,i0,j1,i1]*B[i1,p,i2]) * conj(A)[j1,p,j2]
end
function absorb_bra_ket(::Val{:left}, T::TMatrix, A::MPSTensor, B::MPSTensor=A)
    @tensor T[j2,i2,j1,i1] := (T[j0,i0,j1,i1]*B[i2,p,i0]) * conj(A)[j2,p,j0]
end

###################### Non-Homomophic Contraction ###################
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
function mpo_ket_prod(O::MPOTensor, B::MPSTensor)
    @tensor T[j1,i1,s,j2,i2] := O[j1,s,c,j2] * B[i1,c,i2]
end
