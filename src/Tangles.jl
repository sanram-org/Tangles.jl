module Tangles

using Networks
import Networks: ImplementorTrait, DelegatorTrait

include("ProductState.jl")
export ProductState, ProductOperator

end
