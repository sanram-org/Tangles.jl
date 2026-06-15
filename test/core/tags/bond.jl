using Test
using Tangles
using Tangles: site, sites, hassite, isbond

test_bond = Bond(site"1", site"2")
@test Tuple(test_bond) == (site"1", site"2")
@test isbond(test_bond)
@test issetequal(sites(test_bond), (site"1", site"2"))
@test collect(test_bond) == [site"1", site"2"]
@test hassite(test_bond, site"1")
@test hassite(test_bond, site"2")

test_bond = bond"1-2"
@test Tuple(test_bond) == (site"1", site"2")
@test isbond(test_bond)
@test issetequal(sites(test_bond), (site"1", site"2"))
@test collect(test_bond) == [site"1", site"2"]
@test hassite(test_bond, site"1")
@test hassite(test_bond, site"2")
@test issetequal(sites(test_bond), (site"1", site"2"))

test_bond = Bond(site"1,2", site"2,1")
@test Tuple(test_bond) == (site"1,2", site"2,1")
@test isbond(test_bond)
@test issetequal(sites(test_bond), (site"1,2", site"2,1"))
@test collect(test_bond) == [site"1,2", site"2,1"]
@test hassite(test_bond, site"1,2")
@test hassite(test_bond, site"2,1")
@test issetequal(sites(test_bond), (site"1,2", site"2,1"))

test_bond = bond"(1,2)-(2,1)"
@test Tuple(test_bond) == (site"1,2", site"2,1")
@test isbond(test_bond)
@test issetequal(sites(test_bond), (site"1,2", site"2,1"))
@test collect(test_bond) == [site"1,2", site"2,1"]
@test hassite(test_bond, site"1,2")
@test hassite(test_bond, site"2,1")
@test issetequal(sites(test_bond), (site"1,2", site"2,1"))

ab = bond"1-2"
ba = bond"2-1"
@test isequal(ab, ba)
@test hash(ab, zero(UInt)) == hash(ba, zero(UInt))
@test Set([ab, ba]) == Set([ab]) == Set([ba])
@test isequal(Bond(site"1", site"2"), Bond(site"2", site"1"))
@test isequal(Bond(site"1", site"2"), bond"1-2")

# b = bond"1 | :ket"
# @test b == BoundaryBond(site"1", :ket)
# @test site(b) == site"1"
# @test sites(b) == (site"1",)
# @test boundary(b) == :ket
# @test isequal(b, BoundaryBond(site"1", :ket))
# @test hash(b, zero(UInt)) == hash(BoundaryBond(site"1", :ket))

# # test interpolation for boundary
# bound = :ket
# b = bond"1 | $bound"
# @test b == BoundaryBond(site"1", bound)
# @test boundary(b) == bound

# bound = site"2"
# b = bond"1 | $bound"
# @test b == BoundaryBond(site"1", bound)
# @test boundary(b) == bound
