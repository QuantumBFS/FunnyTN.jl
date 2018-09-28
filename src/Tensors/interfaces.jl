const ↑ = LegIndex{:up}()
const ↓ = LegIndex{:down}()
const → = LegIndex{:last}()
const ← = LegIndex{:first}()

############### contraction #################
∾(l1::Leg, l2::Leg) = glue(l1, l2)
∘(ts1::Tensor, ts2::Tensor) = chain_tensors(ts1, ts2)

⧷(t::Tensor, axis::Int) = Leg(t, axis)
