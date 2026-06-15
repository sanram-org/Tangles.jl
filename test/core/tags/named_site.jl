using Test
using Tangles
using Tangles: NamedSite

x = NamedSite("a")
@test string(x) == "a"

x = NamedSite(:a)
@test string(x) == "a"

@testset "isequal" begin
    @test isequal(NamedSite("a"), NamedSite("a"))
    @test !isequal(NamedSite("a"), NamedSite("b"))
    @test isequal(NamedSite(:a), NamedSite(:a))
    @test !isequal(NamedSite(:a), NamedSite(:b))
    @test !isequal(NamedSite(:a), NamedSite("a"))
end
