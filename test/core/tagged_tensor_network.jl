using Test
using Tangles
using Tangles: LinkBijection, SiteBijection, Site, Link
using DelegatorTraits
using Networks
using Networks: vertex, edge

struct MockSite{S} <: Site
    tag::S
end

Tangles.site(x::MockSite) = site(x.tag)

struct MockLink{L} <: Link
    tag::L
end

Tangles.plug(x::MockLink) = plug(x.tag)

struct WrapperTaggableTensorNetwork{T} <: Tangles.AbstractTensorNetwork
    tn::T
end

Base.copy(tn::WrapperTaggableTensorNetwork) = WrapperTaggableTensorNetwork(copy(tn.tn))
function DelegatorTraits.ImplementorTrait(interface, tn::WrapperTaggableTensorNetwork)
    DelegatorTraits.ImplementorTrait(interface, tn.tn)
end

function DelegatorTraits.DelegatorTrait(interface, tn::WrapperTaggableTensorNetwork)
    if DelegatorTraits.ImplementorTrait(interface, tn.tn) == DelegatorTraits.Implements()
        return DelegateToField{:tn}()
    else
        return DontDelegate()
    end
end

test_tensors = [
    NamedTensor(zeros(2, 2), [Index(:i), Index(:j)]),
    NamedTensor(zeros(2, 3), [Index(:j), Index(:k)]),
    NamedTensor(zeros(2), [Index(:j)]),
]

test_tn = SimpleTensorNetwork(test_tensors)

test_tagged_tn = GenericTensorNetwork(
    test_tn,
    SiteBijection(
        site"1" => vertex_at(test_tn, test_tensors[1]), MockSite(site"2") => vertex_at(test_tn, test_tensors[2])
    ),
    LinkBijection(
        plug"1" => edge_at(test_tn, Index(:i)),
        MockLink(plug"2") => edge_at(test_tn, Index(:k)),
        bond"1-2" => edge_at(test_tn, Index(:j)),
    ),
)

@testset "$(typeof(test_tn))" for test_tn in [test_tagged_tn, WrapperTaggableTensorNetwork(test_tagged_tn)]
    @testset "all_sites" begin
        @test issetequal(all_sites(test_tn), [site"1", MockSite(site"2")])
    end

    @testset "all_links" begin
        @test issetequal(all_links(test_tn), [plug"1", MockLink(plug"2"), bond"1-2"])
    end

    @testset "hassite" begin
        @test hassite(test_tn, site"1")
        @test hassite(test_tn, MockSite(site"2"))
        @test !hassite(test_tn, site"-1")
    end

    @testset "haslink" begin
        @test haslink(test_tn, plug"1")
        @test haslink(test_tn, MockLink(plug"2"))
        @test haslink(test_tn, bond"1-2")
        @test !haslink(test_tn, bond"1-3")
    end

    @testset "nsites" begin
        @test nsites(test_tn) == 2
    end

    @testset "nlinks" begin
        @test nlinks(test_tn) == 3
    end

    @testset "tensor_at(::Site)" begin
        @test tensor_at(test_tn, site"1") === test_tensors[1]
        @test tensor_at(test_tn, MockSite(site"2")) === test_tensors[2]
    end

    @testset "ind_at" begin
        @test ind_at(test_tn, plug"1") == Index(:i)
        @test ind_at(test_tn, MockLink(plug"2")) == Index(:k)
        @test ind_at(test_tn, bond"1-2") == Index(:j)
    end

    @testset "site_at" begin
        @test site_at(test_tn, test_tensors[1]) == site"1"
        @test site_at(test_tn, test_tensors[2]) == MockSite(site"2")
    end

    @testset "link_at" begin
        @test link_at(test_tn, Index(:i)) == plug"1"
        @test link_at(test_tn, Index(:k)) == MockLink(plug"2")
        @test link_at(test_tn, Index(:j)) == bond"1-2"
    end

    @testset "setsite!" begin
        let test_tn = copy(test_tn)
            setsite!(test_tn, test_tensors[3], site"3")
            @test tensor_at(test_tn, site"3") === test_tensors[3]
        end
    end

    @testset "unsetsite!" begin
        let test_tn = copy(test_tn)
            unsetsite!(test_tn, site"1")
            @test !hassite(test_tn, site"1")
        end
    end

    @testset "setlink!" begin end

    @testset "unsetlink!" begin
        let test_tn = copy(test_tn)
            unsetlink!(test_tn, plug"1")
            @test !haslink(test_tn, plug"1")
        end
    end
end
