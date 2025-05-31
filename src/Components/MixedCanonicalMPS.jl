using DelegatorTraits
using Bijections
using ArgCheck

mutable struct MixedCanonicalMatrixProductState <: AbstractMPS
    const tensors::Vector{Tensor}
    orthog_center::MixedCanonical
    const plugs::Bijection{Plug,Index,Dict{Plug,Index},Dict{Index,Plug}}
end

const MixedCanonicalMPS = MixedCanonicalMatrixProductState

function Base.copy(tn::MixedCanonicalMPS)
    MixedCanonicalMPS(copy(tn.tensors), copy(tn.orthog_center), copy(tn.plugs))
end

function MixedCanonicalMPS(arrays; form=MixedCanonical(CartesianSite.(1:length(arrays))), kwargs...)
    MixedCanonicalMPS(form, arrays; kwargs...)
end

function MixedCanonicalMPS(_form::MixedCanonical, arrays; order=defaultorder(MixedCanonicalMPS)) # , check=true)
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MixedCanonicalMPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MixedCanonicalMPS)))"))

    _tensors = Tensor[]
    _plugs = Bijection{Plug,Index}()

    for (i, array) in enumerate(arrays)
        isub = i - 1
        isup = i + 1

        local_order = if i == 1
            filter(x -> x != :l, order)
        elseif i == length(arrays)
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(local_order) do dir
            if dir == :o
                Index(plug"i")
            elseif dir == :r
                Index(bond"i-isup")
            elseif dir == :l
                Index(bond"isub-i")
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end |> collect

        _tensor = Tensor(array, inds)
        push!(_tensors, _tensor)
        _plugs[plug"i"] = Index(plug"i")
    end

    return MixedCanonicalMPS(_tensors, _form, _plugs)
end

# TensorNetwork interface
ImplementorTrait(::TensorNetwork, ::MixedCanonicalMPS) = Implements()

TenetCore.all_tensors(tn::MixedCanonicalMPS) = collect(tn.tensors)
TenetCore.all_tensors_iter(tn::MixedCanonicalMPS) = tn.tensors

TenetCore.tensor_at(tn::MixedCanonicalMPS, site::CartesianSite{1}) = tn.tensors[site.id[1]]
TenetCore.ind_at(tn::MixedCanonicalMPS, plug::Plug) = tn.plugs[plug]

function TenetCore.ind_at(tn::MixedCanonicalMPS, bond::Bond)
    @argcheck hasbond(tn, bond) "Bond $bond not found"
    inds(tensor_at(tn, sites(bond)[1])) ∩ inds(tensor_at(tn, sites(bond)[2])) |> only
end

TenetCore.addtensor!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `addtensor!`")
TenetCore.rmtensor!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `rmtensor!`")

function TenetCore.replace_tensor!(tn::MixedCanonicalMPS, old, new)
    i = findfirst(Base.Fix1(===, old), all_tensors(tn))
    tn.tensors[i] = new
    return tn
end

function TenetCore.replace_ind!(tn::MixedCanonicalMPS, old, new)
    # replace tensors
    for (i, tensor) in enumerate(tn.tensors)
        if old ∈ inds(tensor)
            tn.tensors[i] = replace(tensor, old => new)
        end
    end

    # update plugs
    if hasvalue(tn.plugs, old)
        _plug = inv(tn.plugs)[old]
        tn.plugs[_plug] = new
    end

    return tn
end

# Lattice interface
ImplementorTrait(::TenetCore.Lattice, ::MixedCanonicalMPS) = Implements()

TenetCore.all_sites(tn::MixedCanonicalMPS) = CartesianSite.(1:length(tn.tensors))
TenetCore.all_bonds(tn::MixedCanonicalMPS) = [Bond(CartesianSite.((i, i + 1))...) for i in 1:(length(tn.tensors) - 1)]

function TenetCore.site_at(tn::MixedCanonicalMPS, tensor::Tensor)
    i = findfirst(all_tensors_iter(tn)) do t
        t === tensor
    end
    isnothing(i) && throw(ArgumentError("Tensor not found"))
    return site"i"
end

function TenetCore.bond_at(tn::MixedCanonicalMPS, ind::Index)
    _tensors = tensors_with_inds(tn, ind)
    _sites = site_at.(Ref(tn), _tensors)
    return Bond(_sites...)
end

TenetCore.setsite!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `setsite!`")
TenetCore.setbond!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `setbond!`")
TenetCore.unsetsite!(::MixedCanonicalMPS, site) = error("MixedCanonicalMPS doesn't allow `unsetsite!`")
TenetCore.unsetbond!(::MixedCanonicalMPS, bond) = error("MixedCanonicalMPS doesn't allow `unsetbond!`")

# Pluggable interface
ImplementorTrait(::TenetCore.Pluggable, ::MixedCanonicalMPS) = Implements()

TenetCore.all_plugs(tn::MixedCanonicalMPS) = collect(tn.plugs)
TenetCore.all_plugs_iter(tn::MixedCanonicalMPS) = values(tn.plugs)
TenetCore.hasplug(tn::MixedCanonicalMPS, plug) = haskey(tn.plugs, plug)
TenetCore.nplugs(tn::MixedCanonicalMPS) = length(tn.plugs)

TenetCore.plug_at(tn::MixedCanonicalMPS, ind::Index) = tn.plugs(ind)

TenetCore.setplug!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `setplug!`")
TenetCore.unsetplug!(::MixedCanonicalMPS, args...) = error("MixedCanonicalMPS doesn't allow `unsetplug!`")

# CanonicalForm trait
CanonicalForm(tn::MixedCanonicalMPS) = tn.orthog_center
