module Tangles

using Networks
import Networks: ImplementorTrait, DelegatorTrait

using Reexport
@reexport using TenetCore

include("Interfaces/CanonicalForm.jl")
export NonCanonical, MixedCanonical, BondCanonical, VidalGauge
export form, canonize, canonize!

include("Components/ProductState.jl")
export ProductState, ProductOperator

include("Components/MPO.jl")
export MatrixProductOperator, MPO

include("Components/MPS.jl")
export MatrixProductState, MPS

end
