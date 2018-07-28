# MPS proposal

# 1. Tensor Level Design
## Basic Rules
* take leg, bond, tensor must be at least same priority as `prec-times`

## General Tensor Operations
```julia
t ⧷ 3 # get the third leg of a tensor
```

## Operation on Legs
```julia
l1 ∾ l2 # connect two legs, equivalent to tensordot

l1 ⊶ l2 # left canonical a bond
l1 ⊷ l2 # right canonical a bond

l1 ⪥ l2  # swap two legs, equivalent to swap axes.
(l1, l2, l3) ⪥ (l2, l1, l3)  # permute legs

l1 ⋎ l2 ⋎ l3  # merge legs
l1 ∈ (4, 2)  # split a leg into two of sizes (4, 2)
```

## Examples of Using
1. contract two tensors
```julia
    t3 = t1 ⧷ 3 ∾ t2 ⧷ 1
```
2. multiply a vector on a dimension of tensor
```julia
    (t ⧷ 3) *= v
```

# MPS/MPO Level Design

## MPS/MPO Tensor Contraction
```julia
# t1 ⌶ t2  # transfer matrix, desired but not allowed

t1 ⊂ (t2, [t3])  # contract from left (appear in inner product of MPS)
(t1, [t2]) ⊃ t3  # contract from right

t1 ⪽ (t2, O, [t3])  # contract with operator from left (appear in operator expectation of MPS)
(t1, O, [t2]) ⪾ t3  # contract with operator from right
```

## MPS/MPO Operations
```julia
mps ∘ 5 # 5th tensor, equivalent to mps[5]

# warning: voilation of guide rule!
mps ↑ 5 # 5th upward physical leg
mpo ↓ 5 # 5th downward physical leg of an mpo
mps ⨲ 5 # 5th virtual bond (a bond is a pair of legs)
```

## About Unicode
#### How to type UTF8 characters
https://docs.julialang.org/en/v0.6.0/manual/unicode-input/
#### Supported UTF8 tokens
https://github.com/JuliaLang/julia/blob/c200b4cdb9620b6df369ae3c735cf3af30b6a47f/src/julia-parser.scm
