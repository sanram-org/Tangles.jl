using DelegatorTraits

"""
    Lattice

The `Lattice` interface defines the methods for working with an object with lattice-like structure.
"""
struct Lattice <: Interface end

# keyword-dispatching methods
function sites end
function bonds end

function site end
function bond end

# query methods
function all_sites end
function all_bonds end

function all_sites_iter end
function all_bonds_iter end

function hassite end
function hasbond end

function nsites end
function nbonds end

function site_at end
function bond_at end

function incident_bonds end
function incident_sites end

function neighbor_sites end
function neighbor_bonds end

# mutating methods
function addsite! end
function addbond! end
function rmbond! end
function rmsite! end

function setsite! end
function setbond! end
function unsetsite! end
function unsetbond! end

# optional extra methods
"""
    cart_sites(tn; lt, by, rev, order)

Return a sorted list of `CartesianSite`s in the Tensor Network.

See also: [`all_sites`](@ref)
"""
function cart_sites end

# implementation
## `sites`
# generic implementation for `sites` is type-piracy against QuantumTags
# TODO move `Lattice` interface to its own package and import it both here and in `QuantumTags`
sites(tn::AbstractTensorNetwork; kwargs...) = sites(sort_nt(values(kwargs)), tn)
sites(::@NamedTuple{}, tn) = all_sites(tn)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
# sites(kwargs::NamedTuple{(:like)}, tn) = sites(tn; by=isequal, kwargs...)
# sites(kwargs::NamedTuple{(:by, :like)}, tn) = sites_like(kwargs.by, tn, kwargs.like)

# site(tn; kwargs...) = site(sort_nt(values(kwargs)), tn)
# site(kwargs::NamedTuple{(:at,)}, tn) = site_at(tn, kwargs.at)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
# site(kwargs::NamedTuple{(:like)}, tn) = site(tn; by=isequal, kwargs...)
# site(kwargs::NamedTuple{(:by, :like)}, tn) = site_like(kwargs.by, tn, kwargs.like)

## `bonds`
bonds(tn; kwargs...) = bonds(sort_nt(values(kwargs)), tn)
bonds(::@NamedTuple{}, tn) = all_bonds(tn)

## `site`
# generic implementation for `sites` is type-piracy against QuantumTags
# TODO move `Lattice` interface to its own package and import it both here and in `QuantumTags`
site(tn::AbstractTensorNetwork; kwargs...) = site(sort_nt(values(kwargs)), tn)
site(kwargs::NamedTuple, tn::AbstractTensorNetwork) = only(sites(tn; kwargs...))
site(kwargs::NamedTuple{(:at,)}, tn::AbstractTensorNetwork) = site_at(tn, kwargs.at)

## `bond`
# generic implementation for `sites` is type-piracy against QuantumTags
# TODO move `Lattice` interface to its own package and import it both here and in `QuantumTags`
bond(tn::AbstractTensorNetwork; kwargs...) = bond(sort_nt(values(kwargs)), tn)
bond(kwargs::NamedTuple, tn::AbstractTensorNetwork) = only(bonds(tn; kwargs...))
bond(kwargs::NamedTuple{(:at,)}, tn::AbstractTensorNetwork) = bond_at(tn, kwargs.at)

## `all_sites`
all_sites(lattice) = all_sites(lattice, DelegatorTrait(Lattice(), lattice))
all_sites(lattice, ::DelegateToField) = all_sites(delegator(Lattice(), lattice))
all_sites(lattice, ::DontDelegate) = throw(MethodError(all_sites, (lattice,)))

## `all_bonds`
all_bonds(lattice) = all_bonds(lattice, DelegatorTrait(Lattice(), lattice))
all_bonds(lattice, ::DelegateToField) = all_bonds(delegator(Lattice(), lattice))
all_bonds(lattice, ::DontDelegate) = throw(MethodError(all_bonds, (lattice,)))

## `all_sites_iter`
all_sites_iter(lattice) = all_sites_iter(lattice, DelegatorTrait(Lattice(), lattice))
all_sites_iter(lattice, ::DelegateToField) = all_sites_iter(delegator(Lattice(), lattice))
function all_sites_iter(lattice, ::DontDelegate)
    fallback(all_sites_iter)
    all_sites(lattice)
end

## `all_bonds_iter`
all_bonds_iter(lattice) = all_bonds_iter(lattice, DelegatorTrait(Lattice(), lattice))
all_bonds_iter(lattice, ::DelegateToField) = all_bonds_iter(delegator(Lattice(), lattice))
function all_bonds_iter(lattice, ::DontDelegate)
    fallback(all_bonds_iter)
    all_bonds(lattice)
end

## `hassite`
# generic implementation for `sites` is type-piracy against QuantumTags
# TODO move `Lattice` interface to its own package and import it both here and in `QuantumTags`
hassite(lattice::AbstractTensorNetwork, site) = hassite(lattice, site, DelegatorTrait(Lattice(), lattice))
hassite(lattice::AbstractTensorNetwork, site, ::DelegateToField) = hassite(delegator(Lattice(), lattice), site)
function hassite(lattice::AbstractTensorNetwork, site, ::DontDelegate)
    fallback(hassite)
    any(Base.Fix1(is_site_equal, site), all_sites_iter(lattice))
end

## `hasbond`
hasbond(lattice, bond) = hasbond(lattice, bond, DelegatorTrait(Lattice(), lattice))
hasbond(lattice, bond, ::DelegateToField) = hasbond(delegator(Lattice(), lattice), bond)
function hasbond(lattice, bond, ::DontDelegate)
    fallback(hasbond)
    # TODO should we use `==` or sth like `is_bond_equal`?
    # any(Base.Fix1(is_bond_equal, bond), all_bonds_iter(lattice))
    any(==(bond), all_bonds_iter(lattice))
end

## `nsites`
nsites(lattice) = nsites(lattice, DelegatorTrait(Lattice(), lattice))
nsites(lattice, ::DelegateToField) = nsites(delegator(Lattice(), lattice))
function nsites(lattice, ::DontDelegate)
    fallback(nsites)
    all_sites(lattice) |> length
end

## `nbonds`
nbonds(lattice) = nbonds(lattice, DelegatorTrait(Lattice(), lattice))
nbonds(lattice, ::DelegateToField) = nbonds(delegator(Lattice(), lattice))
function nbonds(lattice, ::DontDelegate)
    fallback(nbonds)
    all_bonds(lattice) |> length
end

## `site_at`
site_at(lattice, tag) = site_at(lattice, tag, DelegatorTrait(Lattice(), lattice))
site_at(lattice, tag, ::DelegateToField) = site_at(delegator(Lattice(), lattice), tag)
site_at(lattice, tag, ::DontDelegate) = throw(MethodError(site_at, (lattice, tag)))

## `bond_at`
bond_at(lattice, tag) = bond_at(lattice, tag, DelegatorTrait(Lattice(), lattice))
bond_at(lattice, tag, ::DelegateToField) = bond_at(delegator(Lattice(), lattice), tag)
bond_at(lattice, tag, ::DontDelegate) = throw(MethodError(bond_at, (lattice, tag)))

## `incident_bonds`
incident_bonds(lattice, site) = incident_bonds(lattice, site, DelegatorTrait(Lattice(), lattice))
incident_bonds(lattice, site, ::DelegateToField) = incident_bonds(delegator(Lattice(), lattice), site)
incident_bonds(lattice, site, ::DontDelegate) = throw(MethodError(incident_bonds, (lattice, site)))

## `incident_sites`
incident_sites(lattice, bond) = incident_sites(lattice, bond, DelegatorTrait(Lattice(), lattice))
incident_sites(lattice, bond, ::DelegateToField) = incident_sites(delegator(Lattice(), lattice), bond)
incident_sites(lattice, bond, ::DontDelegate) = throw(MethodError(incident_sites, (lattice, bond)))

## `neighbor_sites`
neighbor_sites(lattice, site) = neighbor_sites(lattice, site, DelegatorTrait(Lattice(), lattice))
neighbor_sites(lattice, site, ::DelegateToField) = neighbor_sites(delegator(Lattice(), lattice), site)
function neighbor_sites(lattice, site, ::DontDelegate)
    fallback(neighbor_sites)
    _bonds = incident_bonds(lattice, site)
    return unique(Iterators.map(b -> only(filter(s -> s != site, sites(b))), _bonds))
end

## `neighbor_bonds`
neighbor_bonds(lattice, bond) = neighbor_bonds(lattice, bond, DelegatorTrait(Lattice(), lattice))
neighbor_bonds(lattice, bond, ::DelegateToField) = neighbor_bonds(delegator(Lattice(), lattice), bond)
function neighbor_bonds(lattice, bond, ::DontDelegate)
    fallback(neighbor_bonds)
    _sites = incident_sites(lattice, bond)
    _neigh_bonds = Iterators.flatmap(_sites) do _site
        filter(s -> !is_bond_equal(s, bond), incident_bonds(lattice, _site))
    end |> collect
    return unique(_neigh_bonds)
end

## `addsite!`
addsite!(lattice, site) = addsite!(lattice, site, DelegatorTrait(Lattice(), lattice))
addsite!(lattice, site, ::DelegateToField) = addsite!(delegator(Lattice(), lattice), site)
addsite!(lattice, site, ::DontDelegate) = throw(MethodError(addsite!, (lattice, site)))

## `addbond!`
addbond!(lattice, bond) = addbond!(lattice, bond, DelegatorTrait(Lattice(), lattice))
addbond!(lattice, bond, ::DelegateToField) = addbond!(delegator(Lattice(), lattice), bond)
addbond!(lattice, bond, ::DontDelegate) = throw(MethodError(addbond!, (lattice, bond)))

## `rmsite!`
rmsite!(lattice, site) = rmsite!(lattice, site, DelegatorTrait(Lattice(), lattice))
rmsite!(lattice, site, ::DelegateToField) = rmsite!(delegator(Lattice(), lattice), site)
rmsite!(lattice, site, ::DontDelegate) = throw(MethodError(rmsite!, (lattice, site)))

## `rmbond!`
rmbond!(lattice, bond) = rmbond!(lattice, bond, DelegatorTrait(Lattice(), lattice))
rmbond!(lattice, bond, ::DelegateToField) = rmbond!(delegator(Lattice(), lattice), bond)
rmbond!(lattice, bond, ::DontDelegate) = throw(MethodError(rmbond!, (lattice, bond)))

## `setsite!`
# TODO check that the site does not exist and that the tensor exists
#   hassite(tn, e.site) && throw(ArgumentError("Lattice already contains site $(e.site)"))
#   hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) does not exist in the lattice"))
setsite!(lattice, vertex, site) = setsite!(lattice, vertex, site, DelegatorTrait(Lattice(), lattice))
setsite!(lattice, vertex, site, ::DelegateToField) = setsite!(delegator(Lattice(), lattice), vertex, site)
setsite!(lattice, vertex, site, ::DontDelegate) = throw(MethodError(setsite!, (lattice, vertex, site)))

## `setbond!`
# TODO check that the bond does not exist and that the tensor exists
#   hasbond(tn, e.bond) && throw(ArgumentError("Lattice already contains bond $(e.bond)"))
#   hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) does not exist in the lattice"))
setbond!(lattice, edge, bond) = setbond!(lattice, edge, bond, DelegatorTrait(Lattice(), lattice))
setbond!(lattice, edge, bond, ::DelegateToField) = setbond!(delegator(Lattice(), lattice), edge, bond)
setbond!(lattice, edge, bond, ::DontDelegate) = thow(MethodError(setbond!, (lattice, edge, bond)))

## `unsetsite!`
# TODO check that the site exists
#   hassite(tn, e.site) || throw(ArgumentError("Lattice does not contain site $(e.site)"))
unsetsite!(lattice, site) = unsetsite!(lattice, site, DelegatorTrait(Lattice(), lattice))
unsetsite!(lattice, site, ::DelegateToField) = unsetsite!(delegator(Lattice(), lattice), site)
unsetsite!(lattice, site, ::DontDelegate) = throw(MethodError(unsetsite!, (lattice, site)))

## `unsetbond!`
# TODO check that the bond exists
#   hasbond(tn, e.bond) || throw(ArgumentError("Lattice does not contain bond $(e.bond)"))
unsetbond!(lattice, bond) = unsetbond!(lattice, bond, DelegatorTrait(Lattice(), lattice))
unsetbond!(lattice, bond, ::DelegateToField) = unsetbond!(delegator(Lattice(), lattice), bond)
unsetbond!(lattice, bond, ::DontDelegate) = throw(MethodError(unsetbond!, (lattice, bond)))

## `cart_sites`
cart_sites(tn; kwargs...) = cart_sites(tn, DelegatorTrait(Lattice(), tn); kwargs...)
cart_sites(tn, ::DelegateToField; kwargs...) = cart_sites(delegator(Lattice(), tn); kwargs...)
function cart_sites(tn, ::DontDelegate; by=site, kwargs...)
    sort!(filter!(s -> site(s) isa CartesianSite, all_sites(tn)); by, kwargs...)
end
