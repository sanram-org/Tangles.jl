using Test
using Tangles
using Tangles: issite

lane = CartesianSite(1)
@test Tuple(lane) == (1,)
@test CartesianIndex(lane) == CartesianIndex(1)
@test ndims(lane) == 1

lane = CartesianSite(1, 2)
@test Tuple(lane) == (1, 2)
@test CartesianIndex(lane) == CartesianIndex((1, 2))
@test ndims(lane) == 2

lane = site"1"
@test Tuple(lane) == (1,)
@test CartesianIndex(lane) == CartesianIndex(1)
@test ndims(lane) == 1

lane = site"1,2"
@test Tuple(lane) == (1, 2)
@test CartesianIndex(lane) == CartesianIndex((1, 2))
@test ndims(lane) == 2

@testset "isless" begin
    @test site"1" < site"2"
    @test site"1,2" < site"1,3"
    @test site"1,2" < site"2,1"

    @test !(site"2" < site"1")
    @test !(site"1,3" < site"1,2")
    @test !(site"2,1" < site"1,2")
end

@testset "arithmetic" begin
    # addition
    @test site"1" + 2 == site"3"
    @test site"1" + (2,) == site"3"
    @test 2 + site"1" == site"3"
    @test (2,) + site"1" == site"3"

    @test site"1,2" + (1, 1) == site"2,3"
    @test (1, 1) + site"1,2" == site"2,3"
    @test site"1,2" + site"1,1" == site"2,3"

    # subtraction
    @test site"3" - 2 == site"1"
    @test site"3" - (2,) == site"1"
    @test 3 - site"2" == site"1"
    @test (3,) - site"2" == site"1"

    @test site"2,3" - (1, 1) == site"1,2"
    @test (2, 3) - site"1,1" == site"1,2"
    @test site"2,3" - site"1,1" == site"1,2"
end
