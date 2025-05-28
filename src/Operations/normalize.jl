using DelegatorTraits

struct Normalizable <: Interface end

# TODO this can conflict with `LinearAlgebra.normalize!` so don't export
function normalize! end

normalize!(tn) = normalize!(tn, DelegatorTrait(Normalizable(), tn))
normalize!(tn, ::DelegateTo) = normalize!(delegator(Normalizable(), tn))
normalize!(tn, ::DontDelegate) = throw(MethodError(normalize!, (tn,)))
