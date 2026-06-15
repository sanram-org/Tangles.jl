using MacroTools

# NOTE taken from `set.jl`: this is like `hash` method for `AbstractSet`
const hashs_seed = UInt === UInt64 ? 0x852ada37cfe8e0ce : 0xcfe8e0ce

abstract type Tag end

"""
    Site

Tag abstract type to be associated with `Tensor`s.
"""
abstract type Site <: Tag end

"""
    Link

Tag abstract type to be associated with `Index`s.
"""
abstract type Link <: Tag end

"""
    AbstractBond <: Link

Represents a bond between two [`Site`](@ref) objects.
Any subtype must implement `sites` method.

!!! info

    In order to use `AbstractBond` whithin a set-like context (e.g. as a key in a dictionary), it implements `isequal` and `hash` for set-like equivalence.
    This means that `isequal(bond"1-2", bond"2-1")` and `hash(bond"1-2", bond"2-1")` are `true`, but `bond"1-2" == bond"2-1"` is `false`.
"""
abstract type AbstractBond <: Link end

hassite(bond::AbstractBond, _site) = any(Base.Fix1(isequal, _site), sites(bond))
Core.Tuple(bond::AbstractBond) = Tuple(sites(bond))

# automatic set-like equivalence
function Base.hash(b::AbstractBond, h::UInt)
    hv = hashs_seed
    for _site in sites(b)
        hv ⊻= hash(_site, h)
    end
    hash(hv, h)
end

function Base.isequal(a::AbstractBond, b::AbstractBond)
    s1a, s2a = sites(a)
    s1b, s2b = sites(b)
    isequal(s1a, s1b) && isequal(s2a, s2b) || isequal(s1a, s2b) && isequal(s2a, s1b)
end

Base.IteratorSize(::Type{<:AbstractBond}) = Base.HasLength()
Base.length(bond::AbstractBond) = length(sites(bond))
Base.IteratorEltype(::Type{<:AbstractBond}) = Base.HasEltype()
Base.eltype(bond::AbstractBond) = eltype(sites(bond))
Base.isdone(bond::AbstractBond, state) = isdone(bond, state)

Base.first(bond::AbstractBond) = first(sites(bond))
Base.last(bond::AbstractBond) = last(sites(bond))

Base.getindex(bond::AbstractBond, i) = getindex(sites(bond), i)
Base.iterate(bond::AbstractBond) = iterate(sites(bond))
Base.iterate(bond::AbstractBond, state) = iterate(sites(bond), state)

"""
    AbstractPlug <: Link

Represents a physical index related to a [`Site`](@ref) with a annotation of input or output.
"""
abstract type AbstractPlug <: Link end

@enum PlugKind begin
    PLUG_IN
    PLUG_OUT
    SUPER_PLUG_IN
    SUPER_PLUG_IN_DUAL
    SUPER_PLUG_OUT
    SUPER_PLUG_OUT_DUAL
end

isinput(x::PlugKind) = x ∈ (PLUG_IN, SUPER_PLUG_IN, SUPER_PLUG_IN_DUAL)
isoutput(x::PlugKind) = !isinput(x)

isdual(x::PlugKind) = x ∈ (PLUG_IN, SUPER_PLUG_IN_DUAL, SUPER_PLUG_OUT_DUAL)

abstract type Partition <: Tag end

"""
    CartesianSite(id)
    CartesianSite(i, j, ...)

Represents a physical [`Site`](@ref) in a Cartesian coordinate system.
"""
struct CartesianSite{N} <: Site
    id::NTuple{N,Int}
end

CartesianSite(site::CartesianSite) = site
CartesianSite(id::NTuple{N}) where {N} = CartesianSite{N}(id)
CartesianSite(id::Int) = CartesianSite((id,))
CartesianSite(id::Vararg{Int,N}) where {N} = CartesianSite(id)
CartesianSite(id::Base.CartesianIndex) = CartesianSite(Tuple(id))

Base.show(io::IO, x::CartesianSite) = print(io, "site<$(join(x.id, ','))>")

Base.isless(a::CartesianSite, b::CartesianSite) = a.id < b.id
Base.ndims(::CartesianSite{N}) where {N} = N

Core.Tuple(x::CartesianSite) = x.id
Base.CartesianIndex(x::CartesianSite) = CartesianIndex(Tuple(x))

Base.:(+)(x::CartesianSite{1}, i::Int) = CartesianSite(only(x.id) + i)
Base.:(+)(i::Int, x::CartesianSite{1}) = CartesianSite(i + only(x.id))
Base.:(+)(x::CartesianSite{N}, t::NTuple{N,Int}) where {N} = CartesianSite(Tuple(x) .+ t)
Base.:(+)(t::NTuple{N,Int}, x::CartesianSite{N}) where {N} = CartesianSite(t .+ Tuple(x))
Base.:(+)(x::CartesianSite{N}, y::CartesianSite{N}) where {N} = CartesianSite(Tuple(x) .+ Tuple(y))

Base.:(-)(x::CartesianSite{1}, i::Int) = CartesianSite(only(x.id) - i)
Base.:(-)(i::Int, x::CartesianSite{1}) = CartesianSite(i - only(x.id))
Base.:(-)(x::CartesianSite{N}, t::NTuple{N,Int}) where {N} = CartesianSite(Tuple(x) .- t)
Base.:(-)(t::NTuple{N,Int}, x::CartesianSite{N}) where {N} = CartesianSite(t .- Tuple(x))
Base.:(-)(x::CartesianSite{N}, y::CartesianSite{N}) where {N} = CartesianSite(Tuple(x) .- Tuple(y))

"""
    NamedSite(name)

Represents a site identified by a name. `name` must be a `AbstractString` or `Symbol`.
"""
struct NamedSite{S<:Union{<:AbstractString,Symbol}} <: Site
    id::S
end

Base.string(x::NamedSite) = string(x.id)
Base.show(io::IO, x::NamedSite{<:AbstractString}) = print(io, "site<\"$(x.id)\">")
Base.show(io::IO, x::NamedSite{Symbol}) = print(io, "site<:$(x.id)>")

"""
    Bond(src, dst)

Represents a bond between two [`Site`](@ref) objects.
"""
struct Bond{S<:Site} <: AbstractBond
    sites::NTuple{2,S}
end

Bond(a, b) = Bond((a, b))
hassite(bond::Bond, x) = isequal(bond.sites[1], x) || isequal(bond.sites[2], x)
sites(bond::Bond) = bond.sites

function Base.show(io::IO, x::Bond)
    print(io, "bond<")
    print(io, join(sites(x), " ⟷ "))
    print(io, ">")
end

"""
    Plug(id[; dual = false])
    Plug(i, j, ...[; dual = false])

Represents a physical index related to a [`Site`](@ref) with an annotation of input or output.
"""
Base.@kwdef struct Plug{S<:Site} <: AbstractPlug
    site::S
    isdual::Bool = false
end

Plug(site::S; kwargs...) where {S} = Plug{S}(; site, kwargs...)
Plug(id::Int; kwargs...) = Plug(CartesianSite(id); kwargs...)
Plug(@nospecialize(id::NTuple{N,Int}); kwargs...) where {N} = Plug(CartesianSite(id); kwargs...)
Plug(@nospecialize(id::Vararg{Int,N}); kwargs...) where {N} = Plug(CartesianSite(id); kwargs...)
Plug(@nospecialize(id::CartesianIndex); kwargs...) = Plug(CartesianSite(id); kwargs...)

site(p::Plug) = p.site
isinput(p::Plug) = isdual(p)
isdual(p::Plug) = p.isdual

Base.adjoint(p::Plug) = Plug(site(p); isdual=(!isdual(p)))

function Base.show(io::IO, x::Plug)
    print(io, "plug<")
    print(io, site(x))
    isdual(x) && print(io, "'")
    print(io, ">")
end

"""
    Layer(id)

Represents a partition of [`Site`](@ref)s.
"""
struct Layer{T} <: Partition
    id::T
end

Layer(x::Layer) = x

Base.show(io::IO, x::Layer) = print(io, "layer<$(x.id)>")
Base.show(io::IO, x::Layer{Symbol}) = print(io, "layer<:$(x.id)>")
Base.show(io::IO, x::Layer{<:AbstractString}) = print(io, "layer<\"$(x.id)\">")

"""
    InterLayer(a, b)

Represents a partition between two [`Layer`](@ref)s.
"""
struct InterLayer{A<:Layer,B<:Layer} <: Partition
    src::A
    dst::B
end

InterLayer(x::InterLayer) = x
InterLayer(x::Tuple) = InterLayer(Layer.(x)...)
InterLayer(x::Pair) = InterLayer(Layer(first(x)), Layer(last(x)))
InterLayer(src, dst) = InterLayer(Layer(src), Layer(dst))

layers(x::InterLayer) = (x.src, x.dst)

Base.show(io::IO, x::InterLayer) = print(io, "interlayer<$(x.src) ⟷ $(x.dst)>")

# set-like equivalence for `InterLayer`
function Base.hash(x::InterLayer, h::UInt)
    hv = hashs_seed
    hv ⊻= hash(x.src, h)
    hv ⊻= hash(x.dst, h)
    hash(hv, h)
end

function Base.isequal(a::InterLayer, b::InterLayer)
    isequal(a.src, b.src) && isequal(a.dst, b.dst) || isequal(a.src, b.dst) && isequal(a.dst, b.src)
end

"""
    LayerSite(site, layer)

Represents a [`Site`](@ref) at a specific [`Layer`](@ref).
"""
struct LayerSite{S<:Site,L<:Layer} <: Site
    site::S
    layer::L
end

LayerSite(site, layer) = LayerSite(site, Layer(layer))

site(x::LayerSite) = site(x.site)
layer(x::LayerSite) = layer(partition(x))
partition(x::LayerSite) = x.layer

Base.show(io::IO, x::LayerSite) = print(io, "$(x.site) at $(repr(layer(x)))")

"""
    LayerBond(bond, layer)

Represents a [`Bond`](@ref) at a [`Layer`](@ref).
"""
struct LayerBond{B<:AbstractBond,L<:Layer} <: AbstractBond
    bond::B
    layer::L
end

LayerBond(bond, layer) = LayerBond(bond, Layer(layer))

sites(x::LayerBond) = LayerSite.(sites(x.bond), (x.layer,))
bond(x::LayerBond) = bond(x.bond)
partition(x::LayerBond) = layer(x)
layer(x::LayerBond) = layer(x.layer)

# e.g. a closed plug between two same sites on different layers
"""
    InterLayerBond(bond, interlayer)

Represents a closed [`Plug`](@ref) of same [`Site`](@ref) on different [`Layer`](@ref)s.
"""
struct InterLayerBond{S<:Site,IL<:InterLayer} <: AbstractBond
    site::S
    cut::IL
end

InterLayerBond(site::S, cut::C) where {S<:Site,C} = InterLayerBond(site, InterLayer(cut))

site(x::InterLayerBond) = site(x.site)
sites(x::InterLayerBond) = LayerSite.((site(x),), layers(x.cut))
interlayer(x::InterLayerBond) = x.cut
layers(x::InterLayerBond) = layers(x.cut)

# struct BoundaryBond{S<:Site,B} <: Bond
#     site::S
#     boundary::B
# end

# site(x::BoundaryBond) = x.site
# sites(x::BoundaryBond) = (site(x),)

# boundary(x::BoundaryBond) = x.boundary

# Base.isequal(x::Bond, y::BoundaryBond) = isequal(y, x)
# Base.isequal(x::BoundaryBond, y::Bond) = false
# Base.isequal(x::BoundaryBond, y::BoundaryBond) = isequal(site(x), site(y)) && isequal(x.boundary, y.boundary)

# Base.hash(x::BoundaryBond, h::UInt) = hash((site(x), x.boundary), h)

# function Base.show(io::IO, x::BoundaryBond)
#     print(io, "bond<")
#     print(io, site(x))
#     print(io, " | boundary: ")
#     print(io, x.boundary)
#     print(io, ">")
# end

"""
    LayerPlug(plug, layer)

Represents a [`Plug`](@ref) at a [`Layer`](@ref).
"""
struct LayerPlug{P<:AbstractPlug,L<:Layer} <: AbstractPlug
    plug::P
    layer::L
end

LayerPlug(plug, layer) = LayerPlug(plug, Layer(layer))

site(x::LayerPlug) = LayerSite(site(x.plug), x.layer)
plug(x::LayerPlug) = plug(x.plug)
isdual(x::LayerPlug) = isdual(x.plug)

partition(x::LayerPlug) = layer(x)
layer(x::LayerPlug) = layer(x.layer)

Base.adjoint(x::LayerPlug) = LayerPlug(adjoint(x.plug), layer(x))

# macros
dispatch_site_constructor(x::Site) = x
dispatch_site_constructor(x::Symbol) = NamedSite(x)
dispatch_site_constructor(x::AbstractString) = NamedSite(x)
dispatch_site_constructor(x::Int) = CartesianSite(x)
dispatch_site_constructor(x::NTuple{N,Int}) where {N} = CartesianSite(x)
dispatch_site_constructor(x::Vararg{Int,N}) where {N} = CartesianSite(x)
dispatch_site_constructor(x::Base.CartesianIndex) = CartesianSite(Tuple(x))

function _site_expr(expr)
    expr = MacroTools.postwalk(expr) do x
        Meta.isexpr(x, :$) ? esc(only(x.args)) : x
    end

    return :(dispatch_site_constructor($expr))
end

"""
    site"i,j,..."

Constructs a [`CartesianSite`](@ref) object with the given coordinates. The coordinates are given as a comma-separated list of integers.
"""
macro site_str(str)
    expr = Meta.parse(str)
    _site_expr(expr)
end

dispatch_bond_constructor(a, b) = SimpleBond(a, b)
dispatch_bond_constructor(s) = OpenBond(s)

function _bond_expr(expr)
    # open bond if only one site is given
    if Meta.isexpr(expr, :call) && expr.args[1] == :|
        src, _boundary = expr.args[2:end]
        boundary_expr = MacroTools.postwalk(_boundary) do x
            Meta.isexpr(x, :$) ? esc(only(x.args)) : x
        end
        src_expr = _site_expr(src)
        return :(BoundaryBond($src_expr, $boundary_expr))

    elseif Meta.isexpr(expr, :call) && expr.args[1] == :-
        src, dst = expr.args[2:end]
        src_expr = _site_expr(src)
        dst_expr = _site_expr(dst)
        return :(dispatch_bond_constructor($src_expr, $dst_expr))

    else
        throw(
            ArgumentError(
                "Bond string must be in the form 'src-dst', where src and dst are site strings acceptable for @site_str.",
            ),
        )
    end
end

"""
    bond"i-j"
    bond"(i,j,...)-(k,l,...)"

Constructs a [`SimpleBond`](@ref) object.
[`Site`](@ref)s are given as a comma-separated list of integers, and source and destination sites are separated by a `-`.
"""
macro bond_str(str)
    expr = Meta.parse(str)
    _bond_expr(expr)
end

"""
    plug"i,j,...[']"

Constructs a [`Site`](@ref) object with the given coordinates. The coordinates are given as a comma-separated list of integers.
Optionally, a trailing `'` can be added to indicate that the site is a dual site (i.e. an "input").

See also: [`@site_str`](@ref)
"""
macro plug_str(str)
    isdual = endswith(str, '\'')
    str = chopsuffix(str, "'")
    site_expr = _site_expr(Meta.parse(str))
    return :(SimplePlug($(site_expr); isdual=($isdual)))
end
