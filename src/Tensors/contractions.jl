"""
Tensor Absorption (holomophic transform of tensors)

* absorb_mpo
* absorb_bra_ket
"""
function absorb end

"""
    absorb_mpo({leg::Leg{DIRECTION, T}}, O::MPOTensor) -> T

T' ← (mpo <-> T), returns

T is rank 3, DIRECTION is ↑
           |
        /--O*--∖
     ==    |    ==
        ∖--B---/

T is rank 3, DIRECTION is →
       ‖
     |  |
     B--O*--
     |  |
       ‖
"""
absorb_mpo(O::MPOTensor) = leg -> absorb_mpo(leg, O)
function absorb_mpo(leg::Leg{2, <:MPSTensor}, O::MPOTensor)
    ts = mpo_ket_prod(O, parent(leg))
    reshape(ts, size(ts, 1)*size(ts, 2), size(ts, 3), size(ts, 4)*size(ts, 5))
end
function absorb_mpo(leg::Leg{3, <:Tensor{<:Any, 3}}, O::MPOTensor)
    ts = t3_mpo_prod(parent(leg), O)
    reshape(ts, size(ts, 1)*size(ts, 2), size(ts, 3)*size(ts, 4), size(ts, 5))
end

"""
    absorb_ket({leg::Leg{DIRECTION, T}}, O::MPSTensor) -> T

T' ← (ket <-> T), returns

T is rank 2, DIRECTION is →
       ‖
     |  |
     B--O*--
"""
absorb_ket(K::MPSTensor) = leg -> absorb_ket(leg, K)
function absorb_ket(leg::Leg{2, <:Tensor{<:Any, 2}}, K::MPSTensor)
    ts = t2_mps_prod(parent(leg), K)
    reshape(ts, size(ts, 1)*size(ts, 2), size(ts, 3))
end

"""
    absorb_bra_ket({leg::Leg{DIRECTION, T}}, A::MPSTensor, B::MPSTensor=A) -> Tensor

tensor ← bra, ket -> tensor, leg can be lazy. Returns

T is rank 2, DIRECTION is ⇇
    --A*--⊤
      |   T
    --B---⊥

T is rank 2, DIRECTION is ⇉
    ⊤--A*--
    T  |
    ⊥--B---

T is I, DIRECTION is ⇇
    --A*--⊤
      |   |
    --B---⊥

T is I, DIRECTION is ⇉
    ⊤--A*--
    |  |
    ⊥--B---

T is rank 4, DIRECTION is ⇇
    --A*--⊤--
      |   T
    --B---⊥--

T is rank 4, DIRECTION is ⇉
    --⊤--A*--
      T  |
    --⊥--B---
"""
absorb_bra_ket(A::MPSTensor, B::MPSTensor) = T -> absorb_bra_ket(T, A, B)

function absorb_bra_ket(leg::Leg{:rightright, <:AbstractMatrix}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := (parent(leg)[a, b] * B[b, p, j]) * conj(A[a, p, i])
end
function absorb_bra_ket(::Leg{:rightright, UniformScaling{Bool}}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[i, j] := B[a, p, j] * conj(A[a, p, i])
end
function absorb_bra_ket(leg::Leg{:leftleft, <:AbstractArray}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := (parent(leg)[i, j] * B[b, p, j]) * conj(A[a, p, i])
end
function absorb_bra_ket(::Leg{:leftleft, UniformScaling{Bool}}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[a, b] := B[b, p, i] * conj(A[a, p, i])
end
function absorb_bra_ket(leg::Leg{:rightright, <:TMatrix}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[j0,i0,j2,i2] := (parent(leg)[j0,i0,j1,i1]*B[i1,p,i2]) * conj(A)[j1,p,j2]
end
function absorb_bra_ket(leg::Leg{:leftleft, <:TMatrix}, A::MPSTensor, B::MPSTensor=A)
    @tensor Y[j2,i2,j1,i1] := (parent(leg)[j0,i0,j1,i1]*B[i2,p,i0]) * conj(A)[j2,p,j0]
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

mpo-mps contract, up and down wise, returns

      |
    --O*--
      |
    --B---
"""
function mpo_ket_prod(O::MPOTensor, B::MPSTensor)
    @tensor T[j1,i1,s,j2,i2] := O[j1,s,c,j2] * B[i1,c,i2]
end

"""
    t3_mpo_prod(B::MPSTensor, O::MPOTensor)

tensor(rank=3)-mpo contract, left and right wise, returns

    |  |
    B--O*--
    |  |
"""
function t3_mpo_prod(B::MPSTensor, O::MPOTensor)
    @tensor T[u1,u2,d1,d2,b2] := B[u1,d1,b1] * O[b1,u2,d2,b2]
end

"""
    t2_mps_prod(A::AbstractMatrix, B::MPSTensor) ->

matrix, mps contract, left and right wise, returns

     |  |
     B--O*--
"""
function t2_mps_prod(A::AbstractMatrix, B::MPSTensor)
    @tensor T[u1,u2,b2] := A[u1,b1] * B[b1,u2,b2]
end

"""
    mps_mps_prod(A::MPSTensor, B::MPSTensor)

mps, mps contract, left and right wise, returns

       |  |
     --B--O*--
"""
function mps_mps_prod(A::MPSTensor, B::MPSTensor)
    @tensor T[b0,u1,u2,b2] := A[b0,u1,b1] * B[b1,u2,b2]
end
