using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Tangles
using Documenter

DocMeta.setdocmeta!(Muscle, :DocTestSetup, :(using Tangles); recursive=true)

makedocs(;
    modules=[Tangles],
    authors="Sergio Sánchez Ramírez <sergio.sanchez.ramirez@bsc.es> and contributors",
    repo="https://github.com/bsc-quantic/Tangles.jl/blob/{commit}{path}#{line}",
    sitename="Tangles.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bsc-quantic.github.io/Tangles.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md", "Interfaces" => ["Tensor Network" => "interfaces/tensor-network.md"], "API" => "api.md"
    ],
)

deploydocs(; repo="github.com/bsc-quantic/Tangles.jl", devbranch="master")
