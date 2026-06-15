using Bijections
using Networks
using UUIDs
using DelegatorTraits

"""
    GenericLattice

An object that implements the `Lattice` interface to model arbitrary discrete topologies.
"""
struct GenericLattice
    graph::IncidentNetwork{Networks.Vertex{UUID},Networks.Edge{UUID}}
    sitemap::Bijection{Networks.Vertex{UUID},Site}
    bondmap::Bijection{Networks.Edge{UUID},Bond}
end

function GenericLattice()
    GenericLattice(
        IncidentNetwork{Networks.Vertex{UUID},Networks.Edge{UUID}}(),
        Bijection{Networks.Vertex{UUID},Site}(),
        Bijection{Networks.Edge{UUID},Bond}(),
    )
end

function Base.show(io::IO, g::GenericLattice)
    print(io, "GenericLattice ($(length(g.sitemap)) sites, $(length(g.bondmap)) bonds)")
end

# `Network` interface
DelegatorTraits.DelegatorTrait(::Networks.Network, ::GenericLattice) = DelegatorTraits.DelegateToField{:graph}()

Networks.vertex_at(g::GenericLattice, _site) = g.sitemap(_site)
Networks.edge_at(g::GenericLattice, _bond) = g.bondmap(_bond)

# `Lattice` interface
DelegatorTraits.ImplementorTrait(::Lattice, ::GenericLattice) = DelegatorTraits.Implements()

all_sites(g::GenericLattice) = collect(values(g.sitemap))
all_bonds(g::GenericLattice) = collect(values(g.bondmap))
all_sites_iter(g::GenericLattice) = values(g.sitemap)
all_bonds_iter(g::GenericLattice) = values(g.bondmap)

hassite(g::GenericLattice, _site) = hasvalue(g.sitemap, _site)
hasbond(g::GenericLattice, _bond) = hasvalue(g.bondmap, _bond)
nsites(g::GenericLattice) = length(g.sitemap)
nbonds(g::GenericLattice) = length(g.bondmap)

# TODO change to `incident_edges` on next Networks.jl release
incident_bonds(g::GenericLattice, _site) = bond_at.(Ref(g), vertex_incidents(g, vertex_at(g, _site)))

# TODO change to `incident_vertices` on next Networks.jl release
incident_sites(g::GenericLattice, _bond) = site_at.(Ref(g), edge_incidents(g, edge_at(g, _bond)))

site_at(g::GenericLattice, v::Networks.Vertex) = g.sitemap[v]
bond_at(g::GenericLattice, e::Networks.Edge) = g.bondmap[e]

function addsite!(g::GenericLattice, _site)
    hassite(g, _site) && return g
    v = Networks.Vertex(uuid4())
    addvertex!(g.graph, v)
    g.sitemap[v] = _site

    # TODO change to a "adjacent matrix"-based representation to avoid iterating over all edges
    # for _bond in all_bonds_iter(g)
    #     _sites = sites(_bond)
    #     if _site in _sites
    #         # link the vertex to the bond
    #         e = edge_at(g, _bond)
    #         Networks.link!(g.graph, v, e)
    #     end
    # end

    return g
end

function addbond!(g::GenericLattice, _bond)
    hasbond(g, _bond) && return g
    e = Networks.Edge(uuid4())
    addedge!(g.graph, e)

    # filter to allow for open bonds
    # _sites = filter(s -> hassite(g, s), sites(_bond))
    # @assert !isempty(_sites) "Bond must have at least one site in the lattice"

    for _site in sites(_bond)
        v = vertex_at(g, _site)
        Networks.link!(g.graph, v, e)
    end
    g.bondmap[e] = _bond
    return g
end

# TODO rmsite!, rmbond!

# predefined constructors
# NOTE dynamic-dispatch due to the `Val`-dispatch, but it's ok since will be called direclty by the user on a high level
GenericLattice(kind::Symbol, args...; kwargs...) = GenericLattice(Val(kind), args...; kwargs...)

"""
    GenericLattice(:chain, n; periodic=false)

Create a chain lattice with `n` sites.

!!! warning

    It fails for `periodic=true` and `n <= 2`.
"""
function GenericLattice(::Val{:chain}, n; periodic=false)
    lattice = GenericLattice()
    for i in 1:n
        addsite!(lattice, site"$i")
    end

    for i in 1:(n - 1)
        addbond!(lattice, bond"$i - $(i + 1)")
    end

    if periodic
        addbond!(lattice, bond"1-$n")
    end

    return lattice
end

"""
    GenericLattice(:rectangular, nrows, ncols; periodic=false)

Create a rectangular lattice with `nrows` rows and `ncols` columns.

!!! warning

    It fails for `periodic=true` and `nrows, ncols <= 2`.
"""
function GenericLattice(::Val{:rectangular}, nrows, ncols; periodic=false)
    lattice = GenericLattice()
    for row in 1:nrows, col in 1:ncols
        addsite!(lattice, site"$row,$col")
    end

    for row in 1:nrows, col in 1:(ncols - 1)
        addbond!(lattice, bond"($row,$col) - ($row,$(col + 1))")
    end

    for row in 1:(nrows - 1), col in 1:ncols
        addbond!(lattice, bond"($row,$col) - ($(row + 1),$col)")
    end

    if periodic
        for row in 1:nrows
            addbond!(lattice, bond"($row,1) - ($row,$ncols)")
        end

        for col in 1:ncols
            addbond!(lattice, bond"(1,$col) - ($nrows,$col)")
        end
    end

    return lattice
end

"""
    GenericLattice(:lieb, ncellrows, ncellcols)

Create a Lieb lattice with `ncellrows` cell rows and `ncellcols` cell columns.
"""
function GenericLattice(::Val{:lieb}, ncellrows, ncellcols)
    lattice = GenericLattice()
    nrows, ncols = 1 .+ 2 .* (ncellrows, ncellcols)

    # add vertices
    for row in 1:nrows, col in 1:ncols
        # skip holes
        row % 2 == 0 && col % 2 == 0 && continue
        addsite!(lattice, site"$row,$col")
    end

    # add horizontal edges
    for row in 1:2:nrows, col in 1:(ncols - 1)
        addbond!(lattice, bond"$(row,col) - $(row, col + 1)")
    end

    # add vertical edges
    for row in 1:(nrows - 1), col in 1:2:ncols
        addbond!(lattice, bond"$(row,col) - $(row + 1, col)")
    end

    return lattice
end
