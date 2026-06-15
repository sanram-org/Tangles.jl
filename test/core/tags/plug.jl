using Test
using Tangles
using Tangles: site, isdual, isinput, isoutput

s = Plug(1)
@test site(s) == site"1"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

s = Plug(1; isdual=true)
@test site(s) == site"1"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

s = Plug(1, 2)
@test site(s) == site"1, 2"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

s = Plug(1, 2; isdual=true)
@test site(s) == site"1, 2"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

s = plug"1"
@test site(s) == site"1"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

s = plug"1'"
@test site(s) == site"1"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

s = plug"1,2"
@test site(s) == site"1, 2"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

s = plug"1,2'"
@test site(s) == site"1, 2"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

s = adjoint(plug"1")
@test site(s) == site"1"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

s = adjoint(plug"1'")
@test site(s) == site"1"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

s = adjoint(plug"1,2")
@test site(s) == site"1, 2"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

s = adjoint(plug"1,2'")
@test site(s) == site"1, 2"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

# issue: escaping of `i` var in macro
i = 1
s = plug"$i"
@test site(s) == site"$i"
@test isdual(s) == false
@test !isinput(s)
@test isoutput(s)

s = plug"$i'"
@test site(s) == site"$i"
@test isdual(s) == true
@test isinput(s)
@test !isoutput(s)

@testset "site" begin
    @test isequal(site(plug"1"), site"1")
    @test isequal(site(plug"1"), site(plug"1"))
    @test isequal(site(plug"1"), site(plug"1'"))

    @test isequal(site(plug"1,2"), site"1,2")
    @test isequal(site(plug"1,2"), site(plug"1,2"))
    @test isequal(site(plug"1,2"), site(plug"1,2'"))

    @test !isequal(site(plug"1"), site(plug"2"))
end
