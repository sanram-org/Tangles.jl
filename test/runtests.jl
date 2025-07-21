using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @safetestset "TensorNetwork" include("unit/tensor_network.jl")
    @safetestset "Taggable" include("unit/taggable.jl")
    @safetestset "Pluggable" include("unit/pluggable.jl")
    @safetestset "GenericLattice" include("unit/generic_lattice.jl")
end

# Reactant.jl doesn't support Julia 1.12 yet
if VERSION < v"1.12"
    ENV["TANGLES_TEST_REACTANT"] = "true"
end

@testset "Integration" verbose = true begin
    if !isnothing(get(ENV, "TANGLES_TEST_REACTANT", nothing))
        @safetestset "Reactant" include("integration/reactant.jl")
    end
end
