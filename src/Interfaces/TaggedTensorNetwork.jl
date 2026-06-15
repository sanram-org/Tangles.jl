using DelegatorTraits

"""
    TaggedTensorNetwork

The `TaggedTensorNetwork` interface defines a [`TensorNetwok`](@ref) whose [`Tensor`](@ref NamedTensor)s and [`Index`](@ref)s can be referenced by a more informational [`Tag`](@ref).
"""
struct TaggedTensorNetwork <: Interface end

# keyword-dispatching methods
function sites end
sites(tn::AbstractTensorNetwork; kwargs...) = sites(sort_nt(values(kwargs)), tn)
sites(::@NamedTuple{}, tn) = all_sites(tn)

function site end
site(tn::AbstractTensorNetwork; kwargs...) = site(sort_nt(values(kwargs)), tn)
site(kwargs::NamedTuple, tn::AbstractTensorNetwork) = only(sites(tn; kwargs...))
site(kwargs::NamedTuple{(:at,)}, tn::AbstractTensorNetwork) = site_at(tn, kwargs.at)

function links end
links(tn; kwargs...) = links(sort_nt(values(kwargs)), tn)
links(::@NamedTuple{}, tn) = all_links(tn)

function link end
link(tn; kwargs...) = link(sort_nt(values(kwargs)), tn)
link(kwargs::NamedTuple, tn) = only(links(tn; kwargs...))
link(kwargs::NamedTuple{(:at,)}, tn) = link_at(tn, kwargs.at)

function bonds end
bonds(tn; kwargs...) = bonds(sort_nt(values(kwargs)), tn)
bonds(::@NamedTuple{}, tn) = all_bonds(tn)

function bond end
bond(tn::AbstractTensorNetwork; kwargs...) = bond(sort_nt(values(kwargs)), tn)
bond(kwargs::NamedTuple, tn::AbstractTensorNetwork) = only(bonds(tn; kwargs...))
bond(kwargs::NamedTuple{(:at,)}, tn::AbstractTensorNetwork) = bond_at(tn, kwargs.at)

function plugs end
plugs(tn; kwargs...) = plugs(sort_nt(values(kwargs)), tn)
plugs(::@NamedTuple{}, tn) = all_plugs(tn)
plugs(kwargs::NamedTuple{(:set,)}, tn) = plugs_set(tn, kwargs.set)

plugs_set(tn, set::Symbol) = plugs_set(tn, Val(set))
plugs_set(tn, ::Val{S}) where {S} = throw(ArgumentError("invalid `set` value: $(S)"))

plugs_set(tn, ::Val{:all}) = all_plugs(tn)
plugs_set(tn, ::Val{:in}) = plugs_set_in(tn)
plugs_set(tn, ::Val{:out}) = plugs_set_out(tn)
@deprecate plugs_set(tn, ::Val{:inputs}) plugs(tn; set=:in)
@deprecate plugs_set(tn, ::Val{:outputs}) plugs(tn; set=:out)
plugs_set(tn, ::Val{:dual}) = plugs_set_dual(tn)

function plug end
plug(tn; kwargs...) = plug(sort_nt(values(kwargs)), tn)
plug(kwargs::NamedTuple, tn) = only(plugs(tn, kwargs))
plug(kwargs::NamedTuple{(:at,)}, tn) = plug_at(tn, kwargs.at)

# interface methods
function all_sites end
@delegated interface = TaggedTensorNetwork() all_sites(tn)

function all_links end
@delegated interface = TaggedTensorNetwork() all_links(tn)

function tensor_at end
@delegated interface=TaggedTensorNetwork() tensor_at(tn, site)

function ind_at end
@delegated interface=TaggedTensorNetwork() ind_at(tn, link)

function site_at end
@delegated interface=TaggedTensorNetwork() site_at(tn, tensor)

function link_at end
@delegated interface=TaggedTensorNetwork() link_at(tn, ind)

function all_sites_iter end
@delegated interface = TaggedTensorNetwork() function all_sites_iter(tn)
    fallback(all_sites_iter)
    return all_sites(tn)
end

function all_links_iter end
@delegated interface = TaggedTensorNetwork() function all_links_iter(tn)
    fallback(all_links_iter)
    return all_links(tn)
end

function hassite end
@delegated interface = TaggedTensorNetwork() function hassite(tn, site)
    fallback(hassite)
    return any(Base.Fix1(isequal, site), all_sites_iter(tn))
end

function haslink end
@delegated interface = TaggedTensorNetwork() function haslink(tn, link)
    fallback(haslink)
    return any(Base.Fix1(isequal, link), all_links_iter(tn))
end

function nsites end
@delegated interface = TaggedTensorNetwork() function nsites(tn)
    fallback(nsites)
    return length(all_sites(tn))
end

function nlinks end
@delegated interface = TaggedTensorNetwork() function nlinks(tn)
    fallback(nlinks)
    return length(all_links(tn))
end

function site_incidents end
@delegated interface = TaggedTensorNetwork() function site_incidents(tn, site)
    fallback(site_incidents)
    t = tensor_at(tn, site)
    return map(Base.Fix1(link_at, tn), inds(t))
end

function link_incidents end
@delegated interface = TaggedTensorNetwork() function link_incidents(tn, link)
    fallback(link_incidents)
    ind = ind_at(tn, link)
    return map(Base.Fix1(link_at, tn), tensors_set_contain(tn, ind))
end

function neighbor_sites end
@delegated interface = TaggedTensorNetwork() function neighbor_sites(tn, site)
    fallback(neighbor_sites)
    _bonds = site_incidents(tn, site)
    return unique(Iterators.map(b -> only(filter(s -> s != site, sites(b))), _bonds))
end

function neighbor_links end
@delegated interface = TaggedTensorNetwork() function neighbor_links(tn, link)
    fallback(neighbor_links)
    _sites = link_incidents(tn, link)
    _neigh_links = Iterators.flatmap(_sites) do _site
        filter(s -> !is_link_equal(s, link), site_incidents(tn, _site))
    end |> collect
    return unique(_neigh_links)
end

## bonds & plugs
function all_bonds end
@delegated interface = TaggedTensorNetwork() function all_bonds(tn)
    fallback(all_bonds)
    return filter(isbond, all_links_iter(tn))
end

function all_plugs end
@delegated interface = TaggedTensorNetwork() function all_plugs(tn)
    fallback(all_plugs)
    return filter(isplug, all_links_iter(tn))
end

function all_bonds_iter end
@delegated interface = TaggedTensorNetwork() function all_bonds_iter(tn)
    fallback(all_bonds_iter)
    return Iterators.filter(isbond, all_links_iter(tn))
end

function all_plugs_iter end
@delegated interface = TaggedTensorNetwork() function all_plugs_iter(tn)
    fallback(all_plugs_iter)
    return Iterators.filter(isplug, all_links_iter(tn))
end

function bond_at end
@delegated interface = TaggedTensorNetwork() function bond_at(tn, index)
    fallback(bond_at)
    b = link_at(tn, index)
    @assert isbond(b)
    return b
end

function plug_at end
@delegated interface = TaggedTensorNetwork() function plug_at(tn, index)
    fallback(plug_at)
    p = link_at(tn, index)
    @assert isplug(p)
    return p
end

function hasbond end
@delegated interface = TaggedTensorNetwork() function hasbond(tn, bond)
    fallback(hasbond)
    @assert isbond(bond)
    return haslink(tn, bond)
end

function hasplug end
@delegated interface = TaggedTensorNetwork() function hasplug(tn, plug)
    fallback(hasplug)
    @assert isplug(plug)
    return haslink(tn, plug)
end

function nbonds end
@delegated interface = TaggedTensorNetwork() function nbonds(tn)
    fallback(nbonds)
    return count(isbond, all_links_iter(tn))
end

function nplugs end
@delegated interface = TaggedTensorNetwork() function nplugs(tn)
    fallback(nplugs)
    return count(isplug, all_links_iter(tn))
end

function plugs_set_in end
@delegated interface = TaggedTensorNetwork() function plugs_set_in(tn)
    fallback(plugs_set_in)
    return filter(x -> isinput(x), all_plugs_iter(tn))
end

function plugs_set_out end
@delegated interface = TaggedTensorNetwork() function plugs_set_out(tn)
    fallback(plugs_set_out)
    return filter(x -> isoutput(x), all_plugs_iter(tn))
end

function plugs_set_dual end
@delegated interface = TaggedTensorNetwork() function plugs_set_dual(tn)
    fallback(plugs_set_dual)
    return filter(x -> isdual(x), all_plugs_iter(tn))
end

@delegated interface = TaggedTensorNetwork() function inds_set_physical(tn)
    fallback(inds_set_physical)
    return Index[ind_at(tn, i) for i in all_plugs_iter(tn)]
end

@delegated interface = TaggedTensorNetwork() function inds_set_virtual(tn)
    fallback(inds_set_virtual)
    return setdiff(all_inds(tn), inds_set_physical(tn))
end

@delegated interface = TaggedTensorNetwork() function inds_set_inputs(tn)
    fallback(inds_set_inputs)
    return Index[ind_at(tn, i) for i in plugs_set_in(tn)]
end

@delegated interface = TaggedTensorNetwork() function inds_set_outputs(tn)
    fallback(inds_set_outputs)
    return Index[ind_at(tn, i) for i in plugs_set_out(tn)]
end

@delegated interface = TaggedTensorNetwork() function neighbor_bonds(tn, bond)
    fallback(neighbor_bonds)
    return filter!(isbond, neighbor_links(tn, bond))
end

# mutating methods
function setsite! end
@delegated interface = TaggedTensorNetwork() setsite!(tn, tensor, site)

function setlink! end
@delegated interface = TaggedTensorNetwork() setlink!(tn, index, link)

function unsetsite! end
@delegated interface = TaggedTensorNetwork() unsetsite!(tn, site)

function unsetlink! end
@delegated interface = TaggedTensorNetwork() unsetlink!(tn, link)

# extra methods
"""
    cart_sites(tn; lt, by, rev, order)

Return a sorted list of `CartesianSite`s in the Tensor Network.

See also: [`all_sites`](@ref)
"""
function cart_sites end
@delegated interface = TaggedTensorNetwork() function cart_sites(tn)
    return sort!(filter!(s -> s isa CartesianSite, all_sites(tn)))
end

# dispatches for `inds(tn; set)`
inds_set(tn, ::Val{:physical}) = inds_set_physical(tn)
inds_set(tn, ::Val{:virtual}) = inds_set_virtual(tn)
inds_set(tn, ::Val{:inputs}) = inds_set_in(tn)
inds_set(tn, ::Val{:outputs}) = inds_set_out(tn)

"""
    canonicalize_inds!(tn)

Rename indices in the Tensor Network to `Index(tag)` where `tag` is the `Link` tag.
"""
function canonicalize_inds!(tn)
    for link in all_links_iter(tn)
        replace_ind!(tn, ind_at(tn, link), Index(link))
    end
    return tn
end

"""
    align!(a, ioa, b, iob)

Align the physical indices of `b` to match the physical indices of `a`. `ioa` and `iob` are either `:inputs` or `:outputs`.
"""
function align!(a, ioa, b, iob)
    @assert ioa === :inputs || ioa === :outputs
    @assert iob === :inputs || iob === :outputs

    # If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

    # if !isdisjoint(inds(a), inds(b))
    #     @warn "Overlapping indices"
    # end

    # if reset
    #     @debug "[align!] Renaming indices of b"
    #     resetinds!(b, :gensymclean)
    # end

    target_plugs_a = plugs(a; set=ioa)
    target_plugs_b = plugs(b; set=iob)
    do_dual = ioa == iob ? false : true
    @assert issetequal(target_plugs_a, do_dual ? adjoint.(target_plugs_b) : target_plugs_b)

    replacements = map(target_plugs_a) do plug_a
        plug_b = do_dual ? plug_a' : plug_a
        ind_at(b, plug_b) => ind_at(a, plug_a)
    end |> Dict

    if issetequal(keys(replacements), values(replacements))
        return b
    end

    replace_ind!(b, replacements)

    return a, b
end

align!((a, b)::P) where {P<:Pair} = align!(a, :outputs, b, :inputs)

"""
    @align! a => b reset=true

Rename in-place the indices of the input/output sites of two Pluggable Tensor Networks to be able to connect between them.
"""
macro align!(expr)
    @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    Base.remove_linenums!(expr)
    a, b = expr.args[2:end]

    # @assert Meta.isexpr(reset, :(=)) && reset.args[1] == :reset

    @assert Meta.isexpr(a, :call)
    @assert Meta.isexpr(b, :call)
    ioa, ida = a.args
    iob, idb = b.args
    return quote
        align!($(esc(ida)), $(Meta.quot(ioa)), $(esc(idb)), $(Meta.quot(iob)))
        $(esc(idb))
    end
end

"""
    canconnect(a, b)

Return `true` if two [`TaggedTensorNetwork`](@ref man-interface-pluggable)s can be connected. This means:

 1. The outputs of `a` are a superset of the inputs of `b`.
 2. The outputs of `a` and `b` are disjoint except for the sites that are connected.
"""
function canconnect(a, b)
    return plugs(a; set=:out) ⊇ adjoint.(plugs(b; set=:in)) && isdisjoint(
        setdiff(plugs(a; set=:out), adjoint.(plugs(b; set=:in))),
        setdiff(plugs(b; set=:in), adjoint.(plugs(b; set=:out))),
    )
end

function adjoint_plugs!(tn)
    # update plug information and rename inner indices
    # generate mapping
    mapping = Dict(plug => ind(tn; at=plug) for plug in all_plugs(tn))

    # remove sites preemptively to avoid issues on renaming
    for _plug in all_plugs_iter(tn)
        unsetlink!(tn, _plug)
    end

    # set new site mapping
    for (_plug, index) in mapping
        setlink!(tn, index, _plug')
    end

    # rename inner indices
    # replace!(tn, map(i -> i => Symbol(i, "'"), inds(tn; set=:virtual)))

    return tn
end
