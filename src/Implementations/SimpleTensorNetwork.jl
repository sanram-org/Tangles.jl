using UUIDs
using Networks
using Networks: Vertex, Edge, vertex, edge
using Bijections
using Serialization
using Random

const TensorBijection{V,T} = Bijection{V,T,Dict{V,T},IdDict{T,V}}
const IndexBijection{E,I} = Bijection{E,I,Dict{E,I},Dict{I,E}}

struct SimpleTensorNetwork <: AbstractTensorNetwork
    network::IncidentNetwork{Vertex{UUID},Edge{UUID}}
    tensormap::TensorBijection{Vertex{UUID},NamedTensor}
    indmap::IndexBijection{Edge{UUID},Index}

    unsafe::Ref{Union{Nothing,UnsafeScope}}

    # TODO move to a more open and diverse cache?
    sorted_tensors::CachedField{Vector{NamedTensor}}

    function SimpleTensorNetwork(network, tensormap, indmap; unsafe=nothing, check=isnothing(unsafe))
        sorted_tensors = CachedField{Vector{NamedTensor}}()
        tn = new(network, tensormap, indmap, unsafe, sorted_tensors)

        # check index size consistency if not inside an `UnsafeScope`
        if check && !checksizes(tn)
            throw(DimensionMismatch("Tensor network has inconsistent index sizes"))
        end

        return tn
    end
end

function SimpleTensorNetwork()
    SimpleTensorNetwork(
        IncidentNetwork{Vertex{UUID},Edge{UUID}}(),
        TensorBijection{Vertex{UUID},NamedTensor}(),
        IndexBijection{Edge{UUID},Index}(),
    )
end

# TODO find a way to remove the `unsafe` keyword argument from the constructor
function SimpleTensorNetwork(tensors; unsafe::Union{Nothing,UnsafeScope}=nothing)
    network = IncidentNetwork{Vertex{UUID},Edge{UUID}}()
    tensormap = TensorBijection{Vertex{UUID},NamedTensor}()
    indmap = IndexBijection{Edge{UUID},Index}()

    for tensor in tensors
        # add tensor to the network
        vertex = Vertex(uuid4())
        addvertex!(network, vertex)
        tensormap[vertex] = tensor

        # add indices to the network
        for ind in inds(tensor)
            edge = if !hasvalue(indmap, ind)
                edge = Edge(uuid4())
                addedge!(network, edge)
                indmap[edge] = ind
                edge
            else
                indmap(ind)
            end

            setincident!(network, vertex, edge)
        end
    end

    return SimpleTensorNetwork(network, tensormap, indmap; unsafe=Ref{Union{Nothing,UnsafeScope}}(unsafe))
end

function Base.copy(tn::SimpleTensorNetwork)
    network = copy(tn.network)
    tensormap = copy(tn.tensormap)
    indmap = copy(tn.indmap)
    unsafe = Ref{Union{Nothing,UnsafeScope}}(tn.unsafe[])
    new_tn = SimpleTensorNetwork(network, tensormap, indmap; unsafe)

    # register the new copy to the proper UnsafeScope
    !isnothing(unsafe[]) && push!(unsafe[].refs, WeakRef(new_tn))

    return new_tn
end

# Network interface
DelegatorTrait(::Network, ::SimpleTensorNetwork) = DelegateToField{:network}()

Networks.vertex_at(tn::SimpleTensorNetwork, tensor::NamedTensor) = tn.tensormap(tensor)
Networks.edge_at(tn::SimpleTensorNetwork, index::Index) = tn.indmap(index)

# forbid adding vertices and edges to the network (use `addtensor!` instead)
# TODO use the `IsAllowed` mechanism
Networks.addvertex!(::SimpleTensorNetwork, _) = throw(ErrorException("")) # TODO describe the error
Networks.addedge!(::SimpleTensorNetwork, _) = throw(ErrorException("")) # TODO describe the error
Networks.rmvertex!(::SimpleTensorNetwork, _) = throw(ErrorException("")) # TODO describe the error
Networks.rmedge!(::SimpleTensorNetwork, _) = throw(ErrorException("")) # TODO describe the error
Networks.setincident!(::SimpleTensorNetwork, _, _) = throw(ErrorException("")) # TODO describe the error
Networks.unsetincident!(::SimpleTensorNetwork, _, _) = throw(ErrorException("")) # TODO describe the error

# UnsafeScopeable implementation
ImplementorTrait(::UnsafeScopeable, ::SimpleTensorNetwork) = Implements()

get_unsafe_scope(tn::SimpleTensorNetwork) = tn.unsafe[]
set_unsafe_scope!(tn::SimpleTensorNetwork, uc::Union{Nothing,UnsafeScope}) = tn.unsafe[] = uc

function checksizes(tn::SimpleTensorNetwork)
    for (edge, ind) in tn.indmap
        vertex_set = edge_incidents(tn, edge)
        if !allequal(tensor -> size(tensor, ind), Iterators.map(v -> tn.tensormap[v], vertex_set))
            return false
        end
    end

    return true
end

# TensorNetwork implementation
ImplementorTrait(::TensorNetwork, ::SimpleTensorNetwork) = Implements()

function all_tensors(tn::SimpleTensorNetwork)
    return get!(tn.sorted_tensors) do
        # TODO is okay to use `hash`? we sort to get a "stable" order
        sort!(collect(all_tensors_iter(tn)); by=(x) -> x |> inds .|> hash |> sort)
    end
end

all_tensors_iter(tn::SimpleTensorNetwork) = values(tn.tensormap)

all_inds(tn::SimpleTensorNetwork) = collect(all_inds_iter(tn))
all_inds_iter(tn::SimpleTensorNetwork) = values(tn.indmap)

hastensor(tn::SimpleTensorNetwork, tensor) = hasvalue(tn.tensormap, tensor)
hasind(tn::SimpleTensorNetwork, index) = hasvalue(tn.indmap, index)

ntensors(tn::SimpleTensorNetwork) = length(tn.tensormap)
ninds(tn::SimpleTensorNetwork) = length(tn.indmap)

tensor_at(tn::SimpleTensorNetwork, vertex::Vertex) = tn.tensormap[vertex]
ind_at(tn::SimpleTensorNetwork, edge::Edge) = tn.indmap[edge]

function size_inds(tn::SimpleTensorNetwork)
    return Dict{Index,Int}(index => size(tn, index) for index in all_inds_iter(tn))
end

function size_ind(tn::SimpleTensorNetwork, index::Index)
    vertex_set = edge_incidents(tn, edge_at(tn, index))
    return size(tensor(tn; at=first(vertex_set)), index)
end

function tensors_set_contain(tn::SimpleTensorNetwork, index::Index)
    @assert hasind(tn, index) "index $index not found in tensor network"
    vertex_set = edge_incidents(tn, edge_at(tn, index))
    return collect(
        Iterators.map(vertex_set) do vertex
            tn.tensormap[vertex]
        end,
    )
end

function tensors_set_contain(tn::SimpleTensorNetwork, indices)
    target_tensors = tensors(tn; contain=first(indices))
    filter!(target_tensors) do tensor
        indices ⊆ inds(tensor)
    end
    return target_tensors
end

function addtensor!(tn::SimpleTensorNetwork, tensor::NamedTensor)
    hastensor(tn, tensor) && return tn

    # check index sizes if there isn't an active `UnsafeScope` in the Tensor Network
    if isnothing(get_unsafe_scope(tn))
        for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), inds(tensor) ∩ inds(tn))
            throw(
                DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))")
            )
        end
    end

    # add tensor to the network
    vertex = Vertex(uuid4())
    addvertex!(tn.network, vertex)
    tn.tensormap[vertex] = tensor

    # link vertex with edges
    for ind in inds(tensor)
        target_edge = if !hasvalue(tn.indmap, ind)
            target_edge = Edge(uuid4())
            addedge!(tn.network, target_edge)
            tn.indmap[target_edge] = ind
            target_edge
        else
            edge_at(tn, ind)
        end

        setincident!(tn.network, vertex, target_edge)
    end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

function rmtensor!(tn::SimpleTensorNetwork, tensor::NamedTensor)
    hastensor(tn, tensor) || throw(ArgumentError("Tensor not found"))

    target_vertex = vertex_at(tn, tensor)
    edge_set = vertex_incidents(tn, target_vertex) |> copy

    # remove tensor
    delete!(tn.tensormap, target_vertex)
    rmvertex!(tn.network, target_vertex)

    # NOTE force edge pruning because as of Networks.jl 0.3.0, `EdgePersistence(::IncidentNetwork) = PersistEdges()`
    Networks.prune_edges!(tn.network)

    # remove indices if they were removed
    # TODO maybe we should refactor `rmtensor!` to check if we use a `Network` underneath and then, use the `RemoveVertexEffect` and `RemoveEdgeEffect` effects?
    for edge in edge_set
        if !hasedge(tn, edge)
            delete!(tn.indmap, edge)
        end
    end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

function replace_tensor!(tn::SimpleTensorNetwork, old_tensor, new_tensor)
    hastensor(tn, old_tensor) || throw(ArgumentError("Old tensor not found"))
    old_tensor === new_tensor && return tn
    hastensor(tn, new_tensor) && throw(ArgumentError("New tensor already exists in the network"))

    tn.tensormap[vertex_at(tn, old_tensor)] = new_tensor

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

function replace_ind!(tn::SimpleTensorNetwork, old_index, new_index)
    hasind(tn, old_index) || throw(ArgumentError("Index $old_index not found in tensor network"))
    old_index === new_index && return tn
    hasind(tn, new_index) && throw(ArgumentError("Index $new_index already exists in the network"))

    # replace index
    target_edge = edge_at(tn, old_index)
    tn.indmap[target_edge] = new_index

    # TODO should we move this to the `handle!` method?
    # update indices in involved tensors
    vertex_set = edge_incidents(tn, target_edge)
    for vertex in vertex_set
        old_tensor = tensor_at(tn, vertex)
        new_tensor = replace(old_tensor, old_index => new_index)
        replace_tensor!(tn, old_tensor, new_tensor)
    end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

function replace_ind!(tn::SimpleTensorNetwork, old_new::AbstractDict)
    isdisjoint(values(old_new), inds(tn)) || throw(ArgumentError("New indices must not be already present"))

    # update indices
    for (old_ind, new_ind) in old_new
        _edge = edge_at(tn, old_ind)
        tn.indmap[_edge] = new_ind
    end

    # update tensors
    for tensor in all_tensors(tn)
        if !isdisjoint(inds(tensor), keys(old_new))
            new_inds = [get(old_new, ind, ind) for ind in inds(tensor)]
            new_tensor = NamedTensor(parent(tensor), new_inds)
            replace_tensor!(tn, tensor, new_tensor)
        end
    end

    invalidate!(tn.sorted_tensors)

    return tn
end

# derived methods
Base.:(==)(a::SimpleTensorNetwork, b::SimpleTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::SimpleTensorNetwork, b::SimpleTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

# TODO we need to keep the same tensor vertices... fix serialization in Bijection!
function Serialization.serialize(s::AbstractSerializer, obj::SimpleTensorNetwork)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, SimpleTensorNetwork)
    return serialize(s, all_tensors(obj))
    # TODO fix serialization of tensor tags by storing tensors with a number tag
    # return serialize(s, obj.linkmap)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{SimpleTensorNetwork})
    ts = deserialize(s)
    # linkmap = deserialize(s)
    tn = SimpleTensorNetwork(ts)
    # TODO fix deserialization of tensor tags
    # merge!(tn.linkmap, linkmap)
    return tn
end

"""
    rand(TensorNetwork, n::Integer, regularity::Integer; out = 0, dim = 2:9, seed = nothing, globalind = false)

Generate a random tensor network.

# Arguments

  - `n` Number of tensors.
  - `regularity` Average number of indices per tensor.
  - `out` Number of open indices.
  - `dim` Range of dimension sizes.
  - `seed` If not `nothing`, seed random generator with this value.
  - `globalind` Add a global 'broadcast' dimension to every tensor.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    ::Type{SimpleTensorNetwork},
    n::Integer,
    regularity::Integer;
    out=0,
    dim=2:9,
    seed=nothing,
    globalind=false,
    eltype=Float64,
)
    !isnothing(seed) && Random.seed!(rng, seed)

    inds = letter.(randperm(n * regularity ÷ 2 + out))
    size_dict = Dict(ind => rand(dim) for ind in inds)

    outer_inds = collect(Iterators.take(inds, out))
    inner_inds = collect(Iterators.drop(inds, out))

    candidate_inds = shuffle(
        collect(Iterators.flatten([outer_inds, Iterators.flatten(Iterators.repeated(inner_inds, 2))]))
    )

    inputs = map(x -> [x], Iterators.take(candidate_inds, n))

    for ind in Iterators.drop(candidate_inds, n)
        i = rand(1:n)
        while ind in inputs[i]
            i = rand(1:n)
        end

        push!(inputs[i], ind)
    end

    if globalind
        ninds = length(size_dict)
        ind = letter(ninds + 1)
        size_dict[ind] = rand(dim)
        push!(outer_inds, ind)
        push!.(inputs, (ind,))
    end

    tensors = NamedTensor[NamedTensor(rand(eltype, [size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    return SimpleTensorNetwork(tensors)
end

function Base.rand(::Type{SimpleTensorNetwork}, n::Integer, regularity::Integer; kwargs...)
    return rand(Random.default_rng(), SimpleTensorNetwork, n, regularity; kwargs...)
end

@deprecate(
    Base.rand(rng::Random.AbstractRNG, ::Type{TensorNetwork}, n::Integer, regularity::Integer; kwargs...),
    rand(rng, SimpleTensorNetwork, n, regularity; kwargs...),
    false
)
@deprecate(
    Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer; kwargs...),
    rand(SimpleTensorNetwork, n::Integer, regularity::Integer; kwargs...),
    false
)
