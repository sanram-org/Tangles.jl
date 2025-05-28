using Bijections

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

TenetCore.all_tensors(tn::MixedCanonicalMPS) = tn.tensors

function TenetCore.checkeffect(::MixedCanonicalMPS, e::TenetCore.AddTensorEffect)
    error("MixedCanonicalMPS does not support adding tensors directly")
end

function TenetCore.checkeffect(::MixedCanonicalMPS, e::TenetCore.RemoveTensorEffect)
    error("MixedCanonicalMPS does not support adding tensors directly")
end

function TenetCore.replace_tensor_inner!(tn::MixedCanonicalMPS, old, new)
    i = findfirst(Base.Fix1(===, old), all_tensors(tn))
    tn.tensors[i] = new
    return tn
end

function TenetCore.replace_ind_inner!(tn::MixedCanonicalMPS, old, new)
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
end

# Taggable interface
ImplementorTrait(::TenetCore.Taggable, ::MixedCanonicalMPS) = Implements()

TenetCore.all_sites(tn::MixedCanonicalMPS) = CartesianSite.(1:length(tn.tensors))
function TenetCore.all_links(tn::MixedCanonicalMPS)
    return [Bond(CartesianSite(i), CartesianSite(i + 1)) for i in 1:(length(tn.tensors) - 1)] ∪ plugs(tn)
end

TenetCore.tensor_at(tn::MixedCanonicalMPS, site::CartesianSite{1}) = tn.tensors[site.id[1]]

function TenetCore.ind_at(tn::MixedCanonicalMPS, bond::Bond)
    inds(tensor_at(tn, sites(bond)[1])) ∩ inds(tensor_at(tn, sites(bond)[2])) |> only
end

TenetCore.ind_at(tn::MixedCanonicalMPS, plug::Plug) = tn.plugs[plug]

function TenetCore.checkeffect(::MixedCanonicalMPS, ::TenetCore.TagEffect)
    error("MixedCanonicalMPS does not support tagging operations directly")
end
function TenetCore.checkeffect(::MixedCanonicalMPS, ::TenetCore.UntagEffect)
    error("MixedCanonicalMPS does not support untagging operations directly")
end
function TenetCore.checkeffect(::MixedCanonicalMPS, ::TenetCore.ReplaceEffect{<:Site,<:Site})
    error("MixedCanonicalMPS does not support replacing sites directly")
end
function TenetCore.checkeffect(::MixedCanonicalMPS, ::TenetCore.ReplaceEffect{<:Link,<:Link})
    error("MixedCanonicalMPS does not support replacing plugs directly")
end

# Pluggable interface
ImplementorTrait(::TenetCore.Pluggable, ::MixedCanonicalMPS) = Implements()

TenetCore.all_plugs(tn::MixedCanonicalMPS) = collect(tn.plugs)
TenetCore.all_plugs_iter(tn::MixedCanonicalMPS) = values(tn.plugs)
TenetCore.hasplug(tn::MixedCanonicalMPS, plug) = haskey(tn.plugs, plug)
TenetCore.nplugs(tn::MixedCanonicalMPS) = length(tn.plugs)

function TenetCore.ind_at_plug(tn::MixedCanonicalMPS, plug)
    get(tn.plugs, plug, throw(ArgumentError("Plug not found in MixedCanonicalMPS: $plug")))
end

# Tangle interface
form(tn::MixedCanonicalMPS) = tn.orthog_center

function canonize_inner!(tn::MixedCanonicalMPS, old_form::MixedCanonical, new_form::MixedCanonical)
    old_form == new_form && return tn

    # TODO maybe use sth different to `.id`?
    src_left, src_right = site(min_orthog_center(old_form)).id[1], site(max_orthog_center(old_form)).id[1]
    dst_left, dst_right = site(min_orthog_center(new_form)).id[1] - 1, site(max_orthog_center(new_form)).id[1] + 1

    # left-to-right QR sweep (left-canonical tensors)
    for i in src_left:dst_left
        bond = Bond(CartesianSite(i), CartesianSite(i + 1))
        canonize_site!(tn, site"i", bond; method=:qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in src_right:-1:dst_right
        bond = Bond(CartesianSite(i - 1), CartesianSite(i))
        canonize_site!(tn, site"i", bond; method=:qr)
    end

    tn.orthog_center = copy(new_form)
    return tn
end

function simple_update_inner!(tn::MixedCanonicalMPS, operator::Tensor; kwargs...)
    op_site = unique(site.(plugs(operator)))
    @assert length(op_site) == 2 "Operator must have exactly two sites"

    # move orthogonality center to operator sites
    canonize!(tn, MixedCanonical(op_site))

    # perform the simple update routine
    __simple_update!(tn, operator; kwargs...)

    return tn
end
