using Bijections
using Networks

# TODO use dictionary with parameterized types
const SiteBijection = Bijection{Site,Vertex{UUID},Dict{Site,Vertex{UUID}},Dict{Vertex{UUID},Site}}
const LinkBijection = Bijection{Link,Edge{UUID},Dict{Link,Edge{UUID}},Dict{Edge{UUID},Link}}

struct GenericTensorNetwork <: AbstractTensorNetwork
    tn::SimpleTensorNetwork
    sitemap::SiteBijection
    linkmap::LinkBijection
end

GenericTensorNetwork(; kwargs...) = GenericTensorNetwork(SimpleTensorNetwork(; kwargs...))
GenericTensorNetwork(tn::SimpleTensorNetwork) = GenericTensorNetwork(tn, SiteBijection(), LinkBijection())

# TODO Find a way to remove the `unsafe` keyword argument from the constructor
GenericTensorNetwork(tensors; kwargs...) = GenericTensorNetwork(SimpleTensorNetwork(tensors; kwargs...))

Base.copy(tn::GenericTensorNetwork) = GenericTensorNetwork(copy(tn.tn), copy(tn.sitemap), copy(tn.linkmap))

# Network interface
DelegatorTrait(::Networks.Network, ::GenericTensorNetwork) = DelegateToField{:tn}()

Networks.vertex_at(tn::GenericTensorNetwork, site::Site) = tn.sitemap[site]
Networks.edge_at(tn::GenericTensorNetwork, link::Link) = tn.linkmap[link]

# UnsafeScopeable interface
DelegatorTrait(::UnsafeScopeable, ::GenericTensorNetwork) = DelegateToField{:tn}()

# TensorNetwork interface
DelegatorTrait(::TensorNetwork, ::GenericTensorNetwork) = DelegateToField{:tn}()

tensor_at(tn::GenericTensorNetwork, site::Site) = tensor_at(tn, vertex_at(tn, site))
ind_at(tn::GenericTensorNetwork, link::Link) = ind_at(tn, edge_at(tn, link))

function rmtensor!(tn::GenericTensorNetwork, tensor)
    # it can break the mapping, so untag if the removed tensor is tagged
    _vertex = vertex_at(tn, tensor)
    if hasvalue(tn.sitemap, _vertex)
        site_tag = tn.sitemap(_vertex)
        unsetsite!(tn, site_tag)
    end

    # propagate the mutation
    rmtensor!(tn.tn, tensor)

    return tn
end

function slice!(tn::GenericTensorNetwork, ind, i::Integer)
    # it can break the mapping, so untag if the sliced index is tagged
    _edge = edge_at(tn, ind)
    if hasvalue(tn.linkmap, _edge)
        link_tag = tn.linkmap(_edge)
        unsetlink!(tn, link_tag)
    end

    # propagate the mutation
    slice!(tn.tn, ind, i)

    return tn
end

# TaggedTensorNetwork implementation
ImplementorTrait(::TaggedTensorNetwork, ::GenericTensorNetwork) = Implements()

all_sites(tn::GenericTensorNetwork) = collect(all_sites_iter(tn))
all_links(tn::GenericTensorNetwork) = collect(all_links_iter(tn))

all_sites_iter(tn::GenericTensorNetwork) = keys(tn.sitemap)
all_links_iter(tn::GenericTensorNetwork) = keys(tn.linkmap)

hassite(tn::GenericTensorNetwork, site) = haskey(tn.sitemap, site)
haslink(tn::GenericTensorNetwork, link) = haskey(tn.linkmap, link)

nsites(tn::GenericTensorNetwork) = length(tn.sitemap)
nlinks(tn::GenericTensorNetwork) = length(tn.linkmap)

site_at(tn::GenericTensorNetwork, vertex::Vertex) = tn.sitemap(vertex)
site_at(tn::GenericTensorNetwork, tensor::NamedTensor) = site_at(tn, vertex_at(tn, tensor))

link_at(tn::GenericTensorNetwork, edge::Edge) = tn.linkmap(tn, edge)
link_at(tn::GenericTensorNetwork, ind::Index) = link_at(tn, edge_at(tn, ind))

site_incidents(tn::GenericTensorNetwork, site) = link_at.(Ref(tn), vertex_incidents(tn, vertex_at(tn, site)))
link_incidents(tn::GenericTensorNetwork, link) = site_at.(Ref(tn), edge_incidents(tn, edge_at(tn, link)))

setsite!(tn::GenericTensorNetwork, vertex::Vertex, site) = (tn.sitemap[site]=vertex; tn)
setsite!(tn::GenericTensorNetwork, tensor::NamedTensor, site) = setsite!(tn, vertex_at(tn, tensor), site)

setlink!(tn::GenericTensorNetwork, edge::Edge, link) = (tn.linkmap[link]=edge; tn)
setlink!(tn::GenericTensorNetwork, ind::Index, link) = setlink!(tn, edge_at(tn, ind), link)

unsetsite!(tn::GenericTensorNetwork, site) = (delete!(tn.sitemap, site); tn)
unsetlink!(tn::GenericTensorNetwork, link) = (delete!(tn.linkmap, link); tn)

# derived methods
Base.:(==)(a::GenericTensorNetwork, b::GenericTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::GenericTensorNetwork, b::GenericTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

function Base.rand(::Type{GenericTensorNetwork}, n::Integer, regularity::Integer; kwargs...)
    GenericTensorNetwork(rand(SimpleTensorNetwork, n, regularity; kwargs...))
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GenericTensorNetwork}, n::Integer, regularity::Integer; kwargs...)
    GenericTensorNetwork(rand(rng, SimpleTensorNetwork, n, regularity; kwargs...))
end

function generic_rand_state(lattice::GenericLattice, d, χ; rng=Random.default_rng(), eltype=ComplexF64)
    tn = GenericTensorNetwork()

    for site in all_sites_iter(lattice)
        _site_incidents = site_incidents(lattice, site)
        _inds = [map(Index, _site_incidents); [Index(plug"$site")]]

        array = rand(rng, eltype, fill(χ, length(_site_incidents))..., d)
        tensor = Tensor(array, _inds)

        addtensor!(tn, tensor)
        setsite!(tn, tensor, site)
        setplug!(tn, Index(plug"$site"), plug"$site")
    end

    for bond in all_bonds_iter(lattice)
        setbond!(tn, Index(bond), bond)
    end

    return tn
end

function Base.rand(::Type{GenericTensorNetwork}, lattice::GenericLattice, d, χ; kwargs...)
    generic_rand_state(lattice, d, χ; kwargs...)
end
