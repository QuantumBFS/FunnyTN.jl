include("mps0.jl")

"""
    itebd_halfstep(ΓA, λA, ΓB, λB, U, pars)

Absorb a two-site gate U (not necessarily unitary) into an
MPS defined by ΓA, λA, ΓB, λB, and split the result back
into and MPS of the same form, returning ΓA', λA', ΓB', λB'.
The bond dimension of the MPS is truncated to pars["D"],
where pars is a dictionary.

This is called a "half-step" because we only absorb a U
operating on every second pair of neighbouring sites.
"""
function itebd_halfstep(ΓA, λA, ΓB, λB, U, pars)
    D, d = size(ΓA, 1, 2)
    # The next four lines are equivalent to
    # @tensor lump[x,i,j,y] := (((((diagm(λB)[x,a] * ΓA[a,m,b]) * diagm(λA)[b,c]) * ΓB[c,n,d]) * diagm(λB)[d,y]) * U[m,n,i,j])
    A = ΓA .* reshape(λB, (D,1,1))
    B = ΓB .* reshape(λB, (1,1,D))
    A .*= reshape(λA, (1,1,D))
    @tensor lump[x,i,j,y] := (A[x,m,a] * B[a,n,y]) * U[m,n,i,j]
    lump = reshape(lump, (D*d, d*D))
    ΓA, λA, ΓB = svd(lump)
    ΓA, λA, ΓB = truncate_svd(ΓA, λA, ΓB, pars["D"])
    ΓA = reshape(ΓA, (D, d, D))
    ΓB = reshape(ΓB', (D, d, D))
    λBinv = 1 ./ λB
    # The next two lines are equivalent to
    # @tensor ΓA[x,i,y] := diagm(λBinv)[x,a] * ΓA[a,i,y]
    # @tensor ΓB[x,i,y] := ΓB[x,i,a] * diagm(λBinv)[a,y]
    ΓA .*= reshape(λBinv, (D,1,1))
    ΓB .*= reshape(λBinv, (1,1,D))
    return ΓA, λA, ΓB, λB
end

"""
    itebd_step(ΓA, λA, ΓB, λB, U, pars)

Apply a step of iTEBD into an MPS represented by
ΓA, λA, ΓB, λB, with U being the two-site gate that
defines a layer of (imaginary) time-evolution.
Return a new MPS, ΓA', λA', ΓB', λB'.
See https://arxiv.org/pdf/cond-mat/0605597.pdf,
especially Figure 3. pars is a dictionary of parameters,
that most notably should include the bond dimension
pars["D"] to which the MPS should be truncated.
"""
function itebd_step(ΓA, λA, ΓB, λB, U, pars)
    ΓA, λA, ΓB, λB = itebd_halfstep(ΓA, λA, ΓB, λB, U, pars)
    ΓB, λB, ΓA, λA = itebd_halfstep(ΓB, λB, ΓA, λA, U, pars)
    return ΓA, λA, ΓB, λB
end

"""
    itebd_random_initial(d, D)

Return ΓA, λA, ΓB, λB that define an MPS with two-site
translation invariance in the canonical form, with the
tensor chosen randomly.
"""
function itebd_random_initial(d, D)
    Γ = randn(D, d, D)
    λ = randn(D)
    ΓA, λA, ΓB, λB = double_canonicalize(Γ, λ, Γ, λ)
    return ΓA, λA, ΓB, λB
end

"""
    trotter_gate(h, τ)

Given a two-site gate h (a 4-valent tensor),
return the gate U = e^(-τ h).
"""
function trotter_gate(h, τ)
    d = size(h, 1)
    h = reshape(h, (d*d, d*d))
    U = expm(-τ*h)
    U = reshape(U, (d,d,d,d))
    return U
end

"""
    itebd_optimize(h, pars; evalfunc=nothing)

Apply the iTEBD algorithm to find the ground state of the Hamiltonian
defined by the local Hamiltonian term h. h is assumed to operate on
nearest-neighbours only, and translation invariance is assumed. Return
ΓA, λA, ΓB, λB that define an MPS with two-site translation invariance,
which is guaranteed to be in the canonical form. This MPS approximates
the ground state.
See https://arxiv.org/pdf/cond-mat/0605597.pdf.

pars is a dictionary, where each key-value pair is some parameter
that the algorithm takes. The parameters that should be provided are
"τ_min and τ_step":
    Every time convergence has been reached, the Trotter
    parameter τ is multiplied by τ_step and the optimization
    is restarted, until τ falls below τ_min. τ initially starts
    from 0.1.
"D":
    The bond dimension of the MPS.
"max_iters":
    The maximum number of iTEBD iterations that is done before moving
    on to the next value of τ.
"convergence_eps":
    A threshold for convergence. If the relative difference in the
    vectors of Schmidt values before and after the latest iTEBD
    iteration falls below convergence_eps, we move on to the next value
    of τ.
"inner_iters":
    At every iTEBD iteration, several layers of e^(-τ h) are absorbed
    into the MPS before recanonicalizing and checking for convergence.
    inner_iters specifies how many. Note that the total number of layers
    absorbed during the optimization for a given τ may reach
    inner_iters * max_iters.

evalfunc is an optional function, that should take as arguments
ΓA, λA, ΓB, λB that define the (canonical-form) MPS, and return a string.
This string is then printed after every iTEBD step, in addition to other
information such as the measure of convergence and the current iteration
count. Can be used, for instance, for printing the energy at every
iteration.
"""
function itebd_optimize(h, pars; evalfunc=nothing)
    d = size(h, 1)
    ΓA, λA, ΓB, λB = itebd_random_initial(d, pars["D"])
    τ = 0.1
    while τ > pars["τ_min"]
        @printf("In iTEBD, evolving with τ = %.3e.\n", τ)
        eps = Inf
        counter = 0
        U = trotter_gate(h, τ)
        while eps > pars["convergence_eps"] && counter < pars["max_iters"]
            counter += 1
            old_λA, old_λB = λA, λB
            
            # TODO Create some fancy criterion that determines when we need
            # to recanonicalize.
            for i in 1:pars["inner_iters"]
                ΓA, λA, ΓB, λB = itebd_step(ΓA, λA, ΓB, λB, U, pars)
            end
            ΓA, λA, ΓB, λB = double_canonicalize(ΓA, λA, ΓB, λB)
            eps = vecnorm(old_λA - λA)/vecnorm(λA) + vecnorm(old_λB - λB)/vecnorm(λB)

            @printf("In iTEBD, eps = %.3e, counter = %i", eps, counter)
            if evalfunc != nothing
                evstr = evalfunc(ΓA, λA, ΓB, λB)
                print(evstr)
            end
            println()
        end
        τ *= pars["τ_step"]
    end
    return ΓA, λA, ΓB, λB
end

let
    # A bunch of checks that confirm that double_canonicalize works.
    D = 10
    d = 2
    ΓA, ΓB = randn(D, d, D), randn(D, d, D)
    λA, λB = randn(D), randn(D)
    ΓA, λA, ΓB, λB = double_canonicalize(ΓA, λA, ΓB, λB)
    @tensor should_be_id_Ar[x,y] := ΓA[x,i,a] * ((diagm(λA)[a,b] * conj(diagm(λA))[b,c]) * conj(ΓA)[y,i,c])
    @tensor should_be_id_Br[x,y] := ΓB[x,i,a] * ((diagm(λB)[a,b] * conj(diagm(λB))[b,c]) * conj(ΓB)[y,i,c])
    @tensor should_be_id_Al[x,y] := ΓA[a,i,x] * ((diagm(λB)[a,b] * conj(diagm(λB))[b,c]) * conj(ΓA)[c,i,y])
    @tensor should_be_id_Bl[x,y] := ΓB[a,i,x] * ((diagm(λA)[a,b] * conj(diagm(λA))[b,c]) * conj(ΓB)[c,i,y])
    @show vecnorm(should_be_id_Ar - eye(D,D))
    @show vecnorm(should_be_id_Br - eye(D,D))
    @show vecnorm(should_be_id_Al - eye(D,D))
    @show vecnorm(should_be_id_Bl - eye(D,D))
end

function build_ising_ham(h=1.0)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    I2 = eye(2)
    XX = kron(X, X)
    ZI = kron(Z, I2)
    IZ = kron(I2, Z)
    H = -(XX + h/2*(ZI+IZ))
    return H
end

# Functions for evaluating the ground state energy per site
# (or the expectation of any other two-site operator.)
# dcan stands for "double canonical", meaning the canonical
# form with two-site translation symmetry.

function expect_twositelocal_dcan_AB(ΓA, λA, ΓB, λB, O)
    D = size(ΓA, 1)
    A = reshape(λB, (D,1,1)) .* ΓA .* reshape(λA, (1,1,D))
    B = ΓB .* reshape(λB, (1,1,D))
    @tensor AB[x,i,j,y] := A[x,i,a] * B[a,j,y]
    @tensor expectAB[] := AB[a,i,j,b] * O[i,j,m,n] * conj(AB)[a,m,n,b]
    return expectAB[1]
end

function expect_twositelocal_dcan(ΓA, λA, ΓB, λB, O)
    expectAB = expect_twositelocal_dcan_AB(ΓA, λA, ΓB, λB, O)
    expectBA = expect_twositelocal_dcan_AB(ΓB, λB, ΓA, λA, O)
    expectation = (expectAB + expectBA) / 2.
    return expectation
end

magfield = 1.0
exact_energy = -4/π
h = build_ising_ham(magfield)
h = reshape(h, (2,2,2,2))
pars = Dict(
    "τ_min"  => 5e-4,
    "τ_step" => 1/2,
    "D"      => 70,
    "max_iters"       => 150,
    "convergence_eps" => 1e-6,
    "inner_iters"     => 30
)

# Print energy is a function that takes in ΓA, λA, ΓB, λB,
# evaluates the ground-state energy for h, compares to the
# exact value, and returns a string with this information.
print_energy = (ΓA, λA, ΓB, λB) -> begin
    energy = expect_twositelocal_dcan(ΓA, λA, ΓB, λB, h)
    abs(imag(energy)) > 1e-12 && warn("Imaginary energy value: ", energy)
    energy = real(energy)
    error = abs(energy - exact_energy)/abs(exact_energy)
    str = @sprintf(", energy = %.12e, off by %.3e", energy, error)
end

@time ΓA, λA, ΓB, λB = itebd_optimize(h, pars; evalfunc=print_energy)
;

# Functions for transfer matrices and correlation functions.
# These are like the functions we wrote above for the uniform
# MPS, but now for the two-site translation invariant, canonical
# form MPS ΓA, λA, ΓB, λB.
#
# The code tries to make as much use as possible of the old
# functions for the uniform MPS.

function tm_l_dcan(ΓA, λA, ΓB, λB, x)
    y = tm_l(ΓA, x)
    y = Diagonal(λA) * y * Diagonal(λA)
    y = tm_l(ΓB, y)
    y = Diagonal(λB) * y * Diagonal(λB)
    return y
end

function tm_r_dcan(ΓA, λA, ΓB, λB, x)
    y = Diagonal(λB) * x * Diagonal(λB)
    y = tm_r(ΓB, y)
    y = Diagonal(λA) * y * Diagonal(λA)
    y = tm_r(ΓA, y)
    return y
end

function expect_local_dcan(ΓA, λA, ΓB, λB, O)
    λA_sqr = diagm(λA.^2)
    λB_sqr = diagm(λB.^2)
    expectation_A = expect_local(ΓA, O, λB_sqr, λA_sqr)
    expectation_B = expect_local(ΓB, O, λA_sqr, λB_sqr)
    expectation = (expectation_A + expectation_B)/2.
    return expectation
end

function correlator_twopoint_dcan(ΓA, λA, ΓB, λB, O1, O2, m)
    local_O1 = expect_local_dcan(ΓA, λA, ΓB, λB, O1)
    local_O2 = expect_local_dcan(ΓA, λA, ΓB, λB, O2)
    disconnected = local_O1 * local_O2
    
    l = diagm(λA.^2)
    l = tm_l_op(ΓB, O1, l)
    l = Diagonal(λB) * l * Diagonal(λB)
    
    r = diagm(λB.^2)
    r = tm_r_op(ΓB, O1, r)
    r = Diagonal(λA) * r * Diagonal(λA)
    r = tm_r(ΓA, r)
    
    result = zeros(eltype(ΓA), m)
    result[1] = vec(l)'*vec(r) - disconnected
    for i in 1:m
        r = tm_r_dcan(ΓA, λA, ΓB, λB, r)
        result[i] = vec(l)'*vec(r) - disconnected
    end
    return result
end

using PyPlot

X = [0 1; 1 0]
Z = [1 0; 0 -1]
m = 3000
Zcorrs = correlator_twopoint_dcan(ΓA, λA, ΓB, λB, Z, Z, m)
Xcorrs = correlator_twopoint_dcan(ΓA, λA, ΓB, λB, X, X, m)

xpoints = vcat(1:2:2*m)

# A simple linear fit
n1, n2 = 5, 50  # The points to consider in the fit.
logxsamples = log.(xpoints[n1:n2])
logvaluesamples = log.(abs.(Xcorrs[n1:n2]))
a, b = linreg(logxsamples, logvaluesamples)
@printf("Critical exponent for XX correlator: %.3e (exact value %.3e)", b, -1/4)
fit = [exp(a) * i^b for i in xpoints]

# Behold the polynomial decay of correlators, up to some finite
# distance, after which the nature of the MPS state takes over
# and leads to an exponential dive. Note how different this
# behavior is from the one we observed before for a random MPS.
# For the exact critical ground state, the polynomial decay should
# continue indefinitely.
loglog(xpoints, fit; label="fit")
loglog(xpoints, abs.(Xcorrs); marker="x", ms=1, ls="", label="XX")
loglog(xpoints, abs.(Zcorrs); marker="x", ms=1, ls="", label="ZZ")
legend()

# The same in semilogy, to illustrate the exponential decay
# after a point.
semilogy(xpoints, fit; label="fit")
semilogy(xpoints, abs.(Xcorrs); marker="x", ms=1, ls="", label="XX")
semilogy(xpoints, abs.(Zcorrs); marker="x", ms=1, ls="", label="ZZ")
legend()


