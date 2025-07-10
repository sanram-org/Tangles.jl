using Test
using Tangles
using Tangles: neighbor_sites, neighbor_bonds, incident_bonds, incident_sites

@testset let
    lattice = GenericLattice()

    addsite!(lattice, site"1")
    @test issetequal(all_sites(lattice), [site"1"])
    @test isempty(all_bonds(lattice))

    addsite!(lattice, site"2")
    @test issetequal(all_sites(lattice), [site"1", site"2"])
    @test isempty(all_bonds(lattice))
    @test isempty(neighbor_sites(lattice, site"1"))
    @test isempty(neighbor_sites(lattice, site"2"))
    @test isempty(incident_bonds(lattice, site"1"))
    @test isempty(incident_bonds(lattice, site"2"))

    # closed bond
    addbond!(lattice, bond"1-2")
    @test issetequal(all_sites(lattice), [site"1", site"2"])
    @test issetequal(all_bonds(lattice), [bond"1-2"])
    @test issetequal(neighbor_sites(lattice, site"1"), [site"2"])
    @test issetequal(neighbor_sites(lattice, site"2"), [site"1"])
    @test isempty(neighbor_bonds(lattice, bond"1-2"))
    @test issetequal(incident_bonds(lattice, site"1"), [bond"1-2"])
    @test issetequal(incident_bonds(lattice, site"2"), [bond"1-2"])
    @test issetequal(incident_sites(lattice, bond"1-2"), [site"1", site"2"])

    # NOTE forbidden for the time being
    # open bond
    # addbond!(lattice, bond"2-3")
    # @test issetequal(all_sites(lattice), [site"1", site"2"])
    # @test issetequal(all_bonds(lattice), [bond"1-2", bond"2-3"])
    # @test issetequal(neighbor_sites(lattice, site"1"), [site"2"])
    # @test issetequal(neighbor_sites(lattice, site"2"), [site"1"])
    # @test issetequal(neighbor_bonds(lattice, bond"1-2"), [bond"2-3"])
    # @test issetequal(incident_bonds(lattice, site"1"), [bond"1-2"])
    # @test issetequal(incident_bonds(lattice, site"2"), [bond"1-2", bond"2-3"])
    # @test issetequal(incident_sites(lattice, bond"1-2"), [site"1", site"2"])
    # @test issetequal(incident_sites(lattice, bond"2-3"), [site"2"])
end

@testset "constructor: chain - open" begin
    lattice = GenericLattice(:chain, 5)
    @test issetequal(all_sites(lattice), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(lattice), [bond"1-2", bond"2-3", bond"3-4", bond"4-5"])
end

@testset "constructor: chain - periodic" begin
    lattice = GenericLattice(:chain, 5; periodic=true)
    @test issetequal(all_sites(lattice), [site"1", site"2", site"3", site"4", site"5"])
    @test issetequal(all_bonds(lattice), [bond"1-2", bond"2-3", bond"3-4", bond"4-5", bond"1-5"])
end

@testset "constructor: rectangular - open" begin
    lattice = GenericLattice(:rectangular, 2, 3)
    @test issetequal(all_sites(lattice), [site"1,1", site"1,2", site"1,3", site"2,1", site"2,2", site"2,3"])
    @test issetequal(
        all_bonds(lattice),
        [
            bond"(1,1) - (1,2)",
            bond"(1,2) - (1,3)",
            bond"(2,1) - (2,2)",
            bond"(2,2) - (2,3)",
            bond"(1,1) - (2,1)",
            bond"(1,2) - (2,2)",
            bond"(1,3) - (2,3)",
        ],
    )
end

@testset "constructor: rectangular - periodic" begin
    lattice = GenericLattice(:rectangular, 3, 4; periodic=true)
    @test issetequal(
        all_sites(lattice),
        [
            site"1,1",
            site"1,2",
            site"1,3",
            site"1,4",
            site"2,1",
            site"2,2",
            site"2,3",
            site"2,4",
            site"3,1",
            site"3,2",
            site"3,3",
            site"3,4",
        ],
    )
    @test issetequal(
        all_bonds(lattice),
        [
            # horizontal bonds
            bond"(1,1) - (1,2)",
            bond"(1,2) - (1,3)",
            bond"(1,3) - (1,4)",
            bond"(1,1) - (1,4)",
            bond"(2,1) - (2,2)",
            bond"(2,2) - (2,3)",
            bond"(2,3) - (2,4)",
            bond"(2,1) - (2,4)",
            bond"(3,1) - (3,2)",
            bond"(3,2) - (3,3)",
            bond"(3,3) - (3,4)",
            bond"(3,1) - (3,4)",
            # vertical bonds
            bond"(1,1) - (2,1)",
            bond"(1,2) - (2,2)",
            bond"(1,3) - (2,3)",
            bond"(1,4) - (2,4)",
            bond"(2,1) - (3,1)",
            bond"(2,2) - (3,2)",
            bond"(2,3) - (3,3)",
            bond"(2,4) - (3,4)",
            bond"(1,1) - (3,1)",
            bond"(1,2) - (3,2)",
            bond"(1,3) - (3,3)",
            bond"(1,4) - (3,4)",
        ],
    )
end

@testset "constructor: lieb" begin
    lattice = GenericLattice(:lieb, 1, 2)
    @test issetequal(
        all_sites(lattice),
        [
            site"1,1",
            site"1,2",
            site"1,3",
            site"1,4",
            site"1,5",
            site"2,1",
            site"2,3",
            site"2,5",
            site"3,1",
            site"3,2",
            site"3,3",
            site"3,4",
            site"3,5",
        ],
    )
    @test issetequal(
        all_bonds(lattice),
        [
            bond"(1,1) - (1,2)",
            bond"(1,2) - (1,3)",
            bond"(1,3) - (1,4)",
            bond"(1,4) - (1,5)",
            bond"(3,1) - (3,2)",
            bond"(3,2) - (3,3)",
            bond"(3,3) - (3,4)",
            bond"(3,4) - (3,5)",
            bond"(1,1) - (2,1)",
            bond"(1,3) - (2,3)",
            bond"(1,5) - (2,5)",
            bond"(2,1) - (3,1)",
            bond"(2,3) - (3,3)",
            bond"(2,5) - (3,5)",
        ],
    )
end
