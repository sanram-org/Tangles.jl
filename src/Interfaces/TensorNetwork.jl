using DelegatorTraits
using Base: AbstractVecOrTuple
using Networks
using Muscle: Muscle

# interface object
"""
    TensorNetwork <: Interface

A singleton type that represents the basic interface of a Tensor Network.
A type implementing this interface should also implement the `Network` interface.
"""
struct TensorNetwork <: Interface end

# NOTE do not name it `copy` because it can break calls to `Base.copy`
# function copy_tn end

# query methods
## in reality, the only required methods are `all_*` and the mutating methods
function tensors end
tensors(tn; kwargs...) = tensors(sort_nt(values(kwargs)), tn)
tensors(::@NamedTuple{}, tn) = all_tensors(tn)

# TODO fix grammar error on naming
tensors(kwargs::NamedTuple{(:contain,)}, tn) = tensors_set_contain(tn, kwargs.contain)
tensors(kwargs::NamedTuple{(:intersect,)}, tn) = tensors_set_intersect(tn, kwargs.intersect)
tensors(kwargs::NamedTuple{(:equal,)}, tn) = tensors_set_equal(tn, kwargs.equal)

@deprecate tensors(kwargs::NamedTuple{(:contains,)}, tn) tensors(tn; contain=kwargs.contains)
@deprecate tensors(kwargs::NamedTuple{(:intersects,)}, tn) tensors(tn; intersect=kwargs.intersects)
@deprecate tensors(kwargs::NamedTuple{(:withinds,)}, tn) tensors(tn; equals=kwargs.withinds)

# TODO move `inds` back here
# function inds end # WARN moved to `Operations/AbstractTensorNetwork.jl` to avoid type-piracy

### singular version of `tensors`
function tensor end
tensor(tn; kwargs...) = tensor(sort_nt(values(kwargs)), tn)
tensor(kwargs::NamedTuple, tn) = only(tensors(kwargs, tn))
tensor(kwargs::NamedTuple{(:at,)}, tn) = tensor_at(tn, kwargs.at)

### singular version of `inds`
function ind end
ind(tn; kwargs...) = ind(sort_nt(values(kwargs)), tn)
ind(kwargs::NamedTuple, tn) = only(inds(kwargs, tn))
ind(kwargs::NamedTuple{(:at,)}, tn) = ind_at(tn, kwargs.at)

function inds_set end
inds_set(tn, set::Symbol) = inds_set(tn, Val(set))
inds_set(tn, ::Val{S}) where {S} = throw(ArgumentError("Unknown query: set=$(S)"))
inds_set(tn, ::Val{:all}) = all_inds(tn)
inds_set(tn, ::Val{:open}) = inds_set_open(tn)
inds_set(tn, ::Val{:inner}) = inds_set_inner(tn)
inds_set(tn, ::Val{:hyper}) = inds_set_hyper(tn)

ntensors(tn; kwargs...) = ntensors(sort_nt(values(kwargs)), tn)
ntensors(kwargs::NamedTuple, tn) = length(tensors(kwargs, tn))
### dispatch due to performance reasons: see implementation in src/GenericTensorNetwork.jl
ntensors(::@NamedTuple{}, tn) = ntensors_all(tn)

ninds(tn; kwargs...) = ninds(sort_nt(values(kwargs)), tn)
ninds(kwargs::NamedTuple, tn) = length(inds(kwargs, tn))
### dispatch due to performance reasons: see implementation in src/GenericTensorNetwork.jl
ninds(::@NamedTuple{}, tn) = ninds_all(tn)

# interface methods
function all_tensors end
@delegated interface=TensorNetwork() all_tensors(tn)

function all_inds end
@delegated interface=TensorNetwork() all_inds(tn)

# function all_inds(tn, ::DontDelegate)
#     fallback(all_inds)
#     _inds = Set{Index}()
#     for tensor in all_tensors(tn)
#         for ind in inds(tensor)
#             push!(_inds, ind)
#         end
#     end
#     return collect(_inds)
# end

function all_tensors_iter end
@delegated interface=TensorNetwork() function all_tensors_iter(tn)
    fallback(all_tensors_iter)
    return all_tensors(tn)
end

function all_inds_iter end
@delegated interface=TensorNetwork() function all_inds_iter(tn)
    fallback(all_inds_iter)
    return all_inds(tn)
end

function hastensor end
@delegated interface=TensorNetwork() function hastensor(tn, tensor)
    fallback(hastensor)
    any(Base.Fix1(===, tensor), all_tensors_iter(tn))
end

function hasind end
@delegated interface=TensorNetwork() function hasind(tn, ind)
    fallback(hasind)
    i ∈ all_inds_iter(tn)
end

function ntensors_all end
@delegated interface=TensorNetwork() function ntensors_all(tn)
    fallback(ntensors_all)
    return length(all_tensors(tn))
end

function ninds_all end
@delegated interface=TensorNetwork() function ninds_all(tn)
    fallback(ninds_all)
    return length(all_inds(tn))
end

function tensors_set_equal end
@delegated interface=TensorNetwork() function tensors_set_equal(tn, _inds)
    fallback(tensors_set_equal)
    return filter(t -> issetequal(inds(t), _inds), tensors_set_contain(tn, _inds))
end

function tensors_set_contain end
@delegated interface=TensorNetwork() function tensors_set_contain(tn, _target)
    fallback(tensors_set_contain)
    target = _target isa Index ? [_target] : _target
    return filter(Base.Fix2(⊇, target) ∘ inds, all_tensors_iter(tn))
end

function tensors_set_intersect end
@delegated interface=TensorNetwork() function tensors_set_intersect(tn, _target)
    fallback(tensors_set_intersect)
    target = _target isa Index ? [_target] : _target
    return filter(t -> !isdisjoint(inds(t), target), all_tensors_iter(tn))
end

function inds_set_open end
@delegated interface=TensorNetwork() function inds_set_open(tn)
    fallback(inds_set_open)
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, all_tensors_iter(tn))); init=Dict{Index,Int}())
    append!(selected, Iterators.map(first, Iterators.filter(((k, c),) -> c == 1, histogram)))
    return selected
end

function inds_set_inner end
@delegated interface=TensorNetwork() function inds_set_inner(tn)
    fallback(inds_set_inner)
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, all_tensors_iter(tn))); init=Dict{Index,Int}())
    append!(selected, first.(Iterators.filter(((k, c),) -> c == 2, histogram)))
    return selected
end

function inds_set_hyper end
@delegated interface=TensorNetwork() function inds_set_hyper(tn)
    fallback(inds_set_hyper)
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, all_tensors_iter(tn))); init=Dict{Index,Int}())
    append!(selected, Iterators.map(first, Iterators.filter(((k, c),) -> c >= 3, histogram)))
    return selected
end

function inds_parallel_to end
@delegated interface=TensorNetwork() function inds_parallel_to(tn, parallel_to)
    candidates = filter!(!=(parallel_to), collect(mapreduce(inds, ∩, tensors(tn; contain=parallel_to))))
    return filter(candidates) do i
        length(tensors(tn; contain=i)) == length(tensors(tn; contain=parallel_to))
    end
end

function size_inds end
@delegated interface=TensorNetwork() function size_inds(tn)
    fallback(size_inds)
    sizes = Dict{Index,Int}()
    for ind in all_inds_iter(tn)
        sizes[ind] = size_ind(tensor, ind)
    end
    return sizes
end

function size_ind end
@delegated interface=TensorNetwork() function size_ind(tn, i)
    fallback(size_ind)
    _tensors = tensors_set_contain(tn, i)
    @assert !isempty(_tensors) "Index $i not found in the Tensor Network"
    return size(first(_tensors), i)
end

function tensor_at end
@delegated interface=TensorNetwork() tensor_at(tn, tag)

function ind_at end
@delegated interface=TensorNetwork() ind_at(tn, tag)

## mutating methods
function addtensor! end
@delegated interface=TensorNetwork() addtensor!(tn, tensor)

function rmtensor! end
@delegated interface=TensorNetwork() rmtensor!(tn, tensor)

function replace_tensor! end
@delegated interface=TensorNetwork() replace_tensor!(tn, old, new)

function replace_ind! end
@delegated interface=TensorNetwork() replace_ind!(tn, old, new)

"""
    slice!(tn, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice! end

# TODO move to SimpleNetwork
@delegated interface=TensorNetwork() function slice!(tn, ind, i)
    fallback(slice!)
    hasind(tn, ind) || throw(ArgumentError("Index $ind not found in tensor network"))
    target_edge = edge_at(tn, ind)

    # update tensors
    for old_tensor in tensors(tn; contain=ind)
        new_tensor = selectdim(old_tensor, ind, i)
        replace_tensor!(tn, old_tensor, new_tensor)
    end

    # update network: if `i` is an integer, the index disappears and the edge is removed
    if i isa Integer
        rmedge!(tn.network, target_edge)
        delete!(tn.indmap, target_edge)
    end

    return tn
end

"""
    fuse!(tn, ind)

Group indices parallel to `ind` and reshape the tensors accordingly.
"""
function fuse! end

# TODO move implementation to SimpleTensorNetwork?
# TODO replace ind for `Index(Fused(parinds))`?
# TODO should this be run on the lowest or the highest level of the delegation hierarchy?
@delegated interface=TensorNetwork() function fuse!(tn, i)
    fallback(fuse!)
    @assert hasind(tn, i) "Index $i not found in the Tensor Network"

    parinds = inds(tn; parallelto=i)
    length(parinds) == 0 && return tn

    parinds = (i,) ∪ parinds
    @unsafe_region tn for tensor in tensors(tn; intersect=parinds)
        # TODO maybe refactor this when we stop using `Tensors` as graph vertices?
        replace_tensor!(tn, tensor, fuse(tensor, parinds))
    end
    return tn
end

# TODO contract!, split!
