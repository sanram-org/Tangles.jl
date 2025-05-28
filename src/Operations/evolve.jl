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
