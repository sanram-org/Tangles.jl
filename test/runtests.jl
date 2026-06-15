using Test
using SafeTestsets

target_testsets = isempty(ARGS) ? ["core", "integration"] : ARGS

if "core" in target_testsets
    @testset "Unit" verbose = true begin
        @safetestset "TensorNetwork" include("unit/tensor_network.jl")
        @safetestset "Taggable" include("unit/taggable.jl")
        @safetestset "Pluggable" include("unit/pluggable.jl")
        @safetestset "GenericLattice" include("unit/generic_lattice.jl")
        @safetestset "LayeredTensorNetwork" include("unit/layered_tensor_network.jl")
    end
end

if "integration" in target_testsets
    @testset "Integration" verbose = true begin
        if !isnothing(get(ENV, "TANGLES_TEST_REACTANT", nothing))
            @safetestset "Reactant" include("integration/reactant.jl")
        end
    end
end
