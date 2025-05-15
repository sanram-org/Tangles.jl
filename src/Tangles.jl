module Tangles

using Networks
import Networks: ImplementorTrait, DelegatorTrait

using Reexport
@reexport using TenetCore

include("ProductState.jl")
export ProductState, ProductOperator

end
