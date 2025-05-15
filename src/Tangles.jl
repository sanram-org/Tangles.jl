module Tangles

using Networks
import Networks: ImplementorTrait, DelegatorTrait

using Reexport
@reexport using TenetCore

include("Interfaces/CanonicalForm.jl")
export NonCanonical, MixedCanonical, BondCanonical, VidalGauge
export form, canonize, canonize!

include("ProductState.jl")
export ProductState, ProductOperator

include("MPO.jl")
export MatrixProductOperator, MPO

include("MPS.jl")
export MatrixProductState, MPS

end
