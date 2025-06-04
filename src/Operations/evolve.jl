using DelegatorTraits

struct Evolvable <: Interface end

# trait
abstract type EvolveAlgorithm end
struct UnknownAlgorithm <: EvolveAlgorithm end
struct SimpleUpdate <: EvolveAlgorithm end

function evolve! end

evolve!(tn, op; kwargs...) = evolve!(tn, op, DelegatorTrait(Evolvable(), tn); kwargs...)
evolve!(tn, op, ::DelegateToField; kwargs...) = evolve!(delegator(Evolvable(), tn), op; kwargs...)
evolve!(tn, op, ::DontDelegate; algorithm=UnknownAlgorithm(), kwargs...) = evolve!(algorithm, tn, op; kwargs...)

# TODO use an `Algorithm` trait to dispatch on the algorithm
function generic_evolve_mps_mpo_direct!(mps, op)
    @argcheck nsites(mps) == nsites(op) "MPS and MPO must have the same number of sites"

    # align MPS and MPO
    op = resetinds!(copy(op))
    align!(mps, :outputs, op, :inputs)

    @unsafe_region mps for i in 1:nsites(mps)
        tensor_mps = tensor_at(mps, site"i")
        tensor_op = tensor_at(op, site"i")
        c = binary_einsum(tensor_mps, tensor_op)
        c = replace(c, ind_at(op, plug"i") => ind_at(mps, plug"i"))

        # fuse virtual indices
        if i > 1
            j = i - 1
            c = Muscle.fuse(c, [ind_at(mps, bond"j-i"), ind_at(op, bond"j-i")]; ind=ind_at(mps, bond"j-i"))
        end
        if i < nsites(mps)
            j = i + 1
            c = Muscle.fuse(c, [ind_at(mps, bond"i-j"), ind_at(op, bond"i-j")]; ind=ind_at(mps, bond"i-j"))
        end

        replace_tensor!(mps, tensor_mps, c)
    end

    return mps
end

evolve!(mps::MPS, op::AbstractMPO) = generic_evolve_mps_mpo_direct!(mps, op)

function evolve!(mps::MixedCanonicalMPS, op::AbstractMPO)
    generic_evolve_mps_mpo_direct!(mps, op)

    # direct method loses canonicity
    canonize!(mps, MixedCanonical(sites(mps)))

    return mps
end
