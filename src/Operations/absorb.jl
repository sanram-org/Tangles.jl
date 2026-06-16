using Muscle

"""
    absorb!(tn, bond=Bond(site1, site2), target_site)

For a given Tensor Network, contract the singular values Λ located in the bond between lanes `site1` and `site2`.

# Keyword arguments

    - `bond` The bond between the singular values tensor and the tensors to be contracted.
    - `dir` The direction of the contraction. Defaults to `:left`.
"""
function absorb! end

absorb(tn, args...; kwargs...) = canonize!(copy(tn), args...; kwargs...)

function absorb!(tn::AbstractMPO, bond, target_site)
    # retrieve Λ tensor
    # TODO fix this or use a VidalLambda site?
    Λ = tensor_at(tn, bond)
    isnothing(Λ) && return tn

    # absorb to the target tensor
    Γ = tensor_at(tn, target_site)
    hadamard!(Γ, Λ)

    # remove Λ from the tensor network
    rmtensor!(tn, Λ)

    return tn
end
