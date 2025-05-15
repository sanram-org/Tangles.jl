module Tangles

using Reexport

@reexport using QuantumTags
@reexport using Muscle
@reexport using Networks
import Networks: ImplementorTrait, DelegatorTrait, Implements, DelegateTo, DontDelegate, Effect, checkeffect, handle!
@reexport using TenetCore

abstract type Tangle <: TenetCore.AbstractTensorNetwork end

include("Interfaces/CanonicalForm.jl")
export NonCanonical, MixedCanonical, BondCanonical, VidalGauge
export form, canonize, canonize!

include("Components/ProductState.jl")
export ProductState, ProductOperator

include("Components/MPO.jl")
export MatrixProductOperator, MPO

include("Components/MPS.jl")
export MatrixProductState, MPS

include("Operations/simple_update.jl")
export simple_update, simple_update!

end
