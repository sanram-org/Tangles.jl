using Test
using SafeTestsets

target_testsets = isempty(ARGS) ? ["core", "integration"] : ARGS

if "core" in target_testsets
    @testset "Core" verbose = true begin
        @testset "Tags" begin
            @safetestset "CartesianSite" include("core/tags/cartesian_site.jl")
            @safetestset "NamedSite" include("core/tags/named_site.jl")
            @safetestset "Bond" include("core/tags/bond.jl")
            @safetestset "Plug" include("core/tags/plug.jl")
            @safetestset "Lambda" include("core/tags/lambda.jl")
            @safetestset "Layer" include("core/tags/layer.jl")
        end
        @safetestset "TensorNetwork" include("core/tensor_network.jl")
        @safetestset "TaggedTensorNetwork" include("core/tagged_tensor_network.jl")
        @safetestset "Pluggable" include("core/pluggable.jl")
        @safetestset "GenericLattice" include("core/generic_lattice.jl")
        @safetestset "LayeredTensorNetwork" include("core/layered_tensor_network.jl")
    end
end

if "integration" in target_testsets
    @testset "Integration" verbose = true begin
        if !isnothing(get(ENV, "TANGLES_TEST_REACTANT", nothing))
            @safetestset "Reactant" include("integration/reactant.jl")
        end
    end
end
