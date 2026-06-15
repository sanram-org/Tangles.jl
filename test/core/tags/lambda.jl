using Test
using QuantumTags

s = LambdaSite(bond"1-2")
@test s isa LambdaSite
@test bond(s) == bond"1-2"
@test sites(s) == sites(bond"1-2")
@test issite(s)

s = lambda"1-2"
@test s isa LambdaSite
@test bond(s) == bond"1-2"
@test sites(s) == sites(bond"1-2")
@test issite(s)

s = lambda"(1,0)-(2,3)"
@test s isa LambdaSite
@test bond(s) == bond"(1,0)-(2,3)"
@test sites(s) == sites(bond"(1,0)-(2,3)")
@test issite(s)

# issue: escaping of `i` var in macro
i = 1
s = lambda"$i-$(i+1)"
@test s isa LambdaSite
@test bond(s) == bond"$i-$(i+1)"
@test sites(s) == sites(bond"$i-$(i+1)")
@test issite(s)

@testset "set-like equivalence" begin
    s1 = lambda"1-2"
    s2 = lambda"2-1"
    @test isequal(s1, s2)
    @test hash(s1, UInt(0)) == hash(s2, UInt(0))
end

@testset "dispatch from `site\"...\"`" begin
    b = bond"1-2"
    s = site"$b"
    @test s isa LambdaSite
    @test bond(s) == b
    @test sites(s) == sites(b)
end
