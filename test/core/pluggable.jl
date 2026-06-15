using Test
using Tangles
using Tangles:
    plugs_set_in,
    plugs_set_out,
    inds_set_physical,
    inds_set_virtual,
    inds_set_in,
    inds_set_out,
    adjoint_plugs!,
    resetinds!,
    align!
using Networks

# 2-site state
fixture1 = let
    a = Tensor(zeros(2, 2), [Index(:i), Index(:j)])
    b = Tensor(zeros(2, 3), [Index(:j), Index(:k)])
    constructor = () -> begin
        tn = GenericTensorNetwork([a, b])

        setlink!(tn, edge_at(tn, Index(:i)), plug"1")
        setlink!(tn, edge_at(tn, Index(:j)), bond"1-2")
        setlink!(tn, edge_at(tn, Index(:k)), plug"2")

        return tn
    end

    (; constructor, all_plugs=[plug"1", plug"2"], plugmap=Dict(plug"1" => Index(:i), plug"2" => Index(:k)))
end

# 2-site operator with input on site 2 and output on site 1
fixture2 = let
    a = Tensor(zeros(2, 2), [Index(:i), Index(:j)])
    b = Tensor(zeros(2, 3), [Index(:j), Index(:k)])
    constructor = () -> begin
        tn = GenericTensorNetwork([a, b])

        setlink!(tn, edge_at(tn, Index(:i)), plug"1")
        setlink!(tn, edge_at(tn, Index(:j)), bond"1-2")
        setlink!(tn, edge_at(tn, Index(:k)), plug"2'")

        return tn
    end

    (; constructor, all_plugs=[plug"1", plug"2'"], plugmap=Dict(plug"1" => Index(:i), plug"2'" => Index(:k)))
end

@testset "all_plugs" begin
    @testset let
        tn = fixture1.constructor()
        @test issetequal(all_plugs(tn), fixture1.all_plugs)
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(all_plugs(tn), fixture2.all_plugs)
    end
end

@testset "hasplug" begin
    @testset let
        tn = fixture1.constructor()
        @test all(p -> hasplug(tn, p), fixture1.all_plugs)
        @test !hasplug(tn, plug"3")
    end

    @testset let
        tn = fixture2.constructor()
        @test all(p -> hasplug(tn, p), fixture2.all_plugs)
        @test !hasplug(tn, plug"3")
    end
end

@testset "nplugs" begin
    @testset let
        tn = fixture1.constructor()
        @test nplugs(tn) == length(fixture1.all_plugs)
    end

    @testset let
        tn = fixture2.constructor()
        @test nplugs(tn) == length(fixture2.all_plugs)
    end
end

@testset "ind_at(::Plug)" begin
    @testset let
        tn = fixture1.constructor()
        @test all(p -> ind_at(tn, p) == fixture1.plugmap[p], fixture1.all_plugs)
        # @test_throws ArgumentError ind_at(tn, plug"3")
    end

    @testset let
        tn = fixture2.constructor()
        @test all(p -> ind_at(tn, p) == fixture2.plugmap[p], fixture2.all_plugs)
        # @test_throws ArgumentError ind_at(tn, plug"3")
    end
end

@testset "plugs_set_in" begin
    @testset let
        tn = fixture1.constructor()
        @test isempty(plugs_set_in(tn))
        @test isempty(plugs(tn; set=:in))
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(plugs_set_in(tn), [plug"2'"])
        @test plugs(tn; set=:in) == plugs_set_in(tn)
    end
end

@testset "plugs_set_out" begin
    @testset let
        tn = fixture1.constructor()
        @test issetequal(plugs_set_out(tn), fixture1.all_plugs)
        @test plugs(tn; set=:out) == plugs_set_out(tn)
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(plugs_set_out(tn), [plug"1"])
        @test plugs(tn; set=:out) == plugs_set_out(tn)
    end
end

@testset "inds_set_physical" begin
    @testset let
        tn = fixture1.constructor()
        @test issetequal(inds_set_physical(tn), [Index(:i), Index(:k)])
        @test inds(tn; set=:physical) == inds_set_physical(tn)
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(inds_set_physical(tn), [Index(:i), Index(:k)])
        @test inds(tn; set=:physical) == inds_set_physical(tn)
    end
end

@testset "inds_set_virtual" begin
    @testset let
        tn = fixture1.constructor()
        @test issetequal(inds_set_virtual(tn), [Index(:j)])
        @test inds(tn; set=:virtual) == inds_set_virtual(tn)
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(inds_set_virtual(tn), [Index(:j)])
        @test inds(tn; set=:virtual) == inds_set_virtual(tn)
    end
end

@testset "inds_set_in" begin
    @testset let
        tn = fixture1.constructor()
        @test isempty(inds_set_in(tn))
        @test isempty(inds(tn; set=:inputs))
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(inds_set_in(tn), [Index(:k)])
        @test inds(tn; set=:inputs) == inds_set_in(tn)
    end
end

@testset "inds_set_out" begin
    @testset let
        tn = fixture1.constructor()
        @test issetequal(inds_set_out(tn), [Index(:i), Index(:k)])
        @test inds(tn; set=:outputs) == inds_set_out(tn)
    end

    @testset let
        tn = fixture2.constructor()
        @test issetequal(inds_set_out(tn), [Index(:i)])
        @test inds(tn; set=:outputs) == inds_set_out(tn)
    end
end

@testset "adjoint_plugs!" begin
    @testset let
        tn = fixture1.constructor()
        adjoint_plugs!(tn)

        @test issetequal(all_plugs(tn), adjoint.(fixture1.all_plugs))
    end

    @testset let
        tn = fixture2.constructor()
        adjoint_plugs!(tn)

        @test issetequal(all_plugs(tn), adjoint.(fixture2.all_plugs))
    end
end

@testset "align!" begin
    @testset let
        ket = fixture1.constructor()
        bra = adjoint_plugs!(resetinds!(copy(ket)))

        align!(ket, :outputs, bra, :inputs)

        @test issetequal(plugs(ket; set=:outputs), adjoint.(plugs(bra; set=:inputs)))
        @test isempty(plugs(ket; set=:inputs))
        @test isempty(plugs(bra; set=:outputs))
    end

    @testset let
        ket = fixture2.constructor()
        bra = adjoint_plugs!(resetinds!(copy(ket)))

        align!(ket, :outputs, bra, :inputs)

        @test issetequal(plugs(ket; set=:outputs), [plug"1"])
        @test issetequal(plugs(bra; set=:inputs), [plug"1'"])
        @test issetequal(plugs(ket; set=:inputs), [plug"2'"])
        @test issetequal(plugs(bra; set=:outputs), [plug"2"])
    end
end
