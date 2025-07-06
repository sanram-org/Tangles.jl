module TanglesReactantExt

using Tangles
using Reactant
using Reactant: Enzyme, @skip_rewrite_func, @skip_rewrite_type
# using PrecompileTools

# issue fix: strange behavior between `IdDict` and `Tensor` when using `TracedRArray`... can't figure out why or do a MWE
function Tangles.hastensor(tn::SimpleTensorNetwork, tensor::Tensor{<:Reactant.TracedRNumber})
    any(t -> tensor === t, all_tensors_iter(tn))
end

function Tangles.vertex_at(tn::SimpleTensorNetwork, tensor::Tensor{<:Reactant.TracedRNumber})
    for (vertex, mapped_tensor) in tn.tensormap
        if mapped_tensor === tensor
            return vertex
        end
    end

    throw(ErrorException("Tensor $tensor not found in tensor network but `checkeffect` should have stopped this!"))
end

# we specify `mode` and `track_numbers` types due to ambiguity
# TODO in Reactant v0.3, rename it to `Reactant.transmute_type`
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:Tangles.AbstractTensorNetwork}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    args...,
)
    return T
end

# TODO replace `tensor_id` for `Vertex{UUID}`?
# TODO in Reactant v0.3, rename it to `Reactant.transmute`
function Reactant.Compiler.make_tracer(seen, prev::SimpleTensorNetwork, @nospecialize(path), mode; kwargs...)
    traced_tn = copy(prev)
    for (i, tensor) in enumerate(all_tensors(traced_tn))
        traced_tensor = Reactant.Compiler.make_tracer(
            seen, tensor, Reactant.append_path(path, (; tensor_id=i)), mode; kwargs...
        )

        # seems like in some tracing modes, the traced tensor is the same as the original tensor
        if tensor !== traced_tensor
            replace_tensor!(traced_tn, tensor, traced_tensor)
        end
    end
    return traced_tn
end

# requires a specialization due to default `create_result` getting confused with `CachedField`
function Reactant.Compiler.create_result(
    tocopy::SimpleTensorNetwork,
    @nospecialize(path),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
)
    expr_network = Reactant.Compiler.create_result(
        tocopy.network,
        Reactant.append_path(path, :network),
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code,
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    expr_indmap = Reactant.Compiler.create_result(
        tocopy.indmap,
        Reactant.append_path(path, :indmap),
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code,
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    # `tensormap` requires special treatment due to the `path` used to store the tensors
    # TODO refactor the way we mark the path of `tensors` in a `AbstractTensorNetwork`
    tensormap_results = map(enumerate(tocopy.tensormap)) do (i, (vertex, tensor))
        :(
            $vertex => $(Reactant.Compiler.create_result(
                tensor,
                Reactant.append_path(path, (; tensor_id=i)),
                result_stores,
                path_to_shard_info,
                to_unreshard_results,
                unresharded_code,
                unresharded_arrays_cache,
                used_shardinfo,
                result_cache,
                var_idx,
                resultgen_code,
            ))
        )
    end
    expr_tensormap = :($(typeof(tocopy.tensormap))([$(tensormap_results...)]))

    return quote
        network = $expr_network
        tensormap = $expr_tensormap
        indmap = $expr_indmap
        $SimpleTensorNetwork(network, tensormap, indmap)
    end
end

Reactant.traced_getfield(x::SimpleTensorNetwork, i::Int) = all_tensors(x)[i]
Reactant.traced_getfield(x::SimpleTensorNetwork, fld::@NamedTuple{tensor_id::Int}) = all_tensors(x)[fld.tensor_id]

function Reactant.TracedUtils.push_val!(ad_inputs, x::SimpleTensorNetwork, path)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1].tensor_id]).mlir_data

    return push!(ad_inputs, x)
end

function Reactant.TracedUtils.set!(x::SimpleTensorNetwork, path, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1].tensor_id])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function Reactant.set_act!(inp::Enzyme.Annotation{SimpleTensorNetwork}, path, reverse, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

    x = if inp isa Enzyme.Active
        inp.val
    else
        inp.dval
    end

    x = parent(tensors(x)[path[1].tensor_id])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

# This function is used to skip rewriting of certain functions and type constructors in Reactant.jl, which is necessary
# for overlaying methods called dynamically. By skipping the rewrite where we know it's ok, Julia compilation should 
# take less time. It must be called on the top level for precompilation and in `__init__` for runtime.
function tangles_skip_rewrites()
    # `TensorNetwork` interface
    @skip_rewrite_type Tangles.tensors
    @skip_rewrite_func Tangles.all_tensors
    @skip_rewrite_func Tangles.all_inds
    @skip_rewrite_func Tangles.hastensor
    @skip_rewrite_func Tangles.hasind
    @skip_rewrite_func Tangles.ntensors
    @skip_rewrite_func Tangles.ninds
    @skip_rewrite_func Tangles.tensors_with_inds
    @skip_rewrite_func Tangles.tensors_contain_inds
    @skip_rewrite_func Tangles.tensors_intersect_inds
    @skip_rewrite_func Tangles.inds_set
    @skip_rewrite_func Tangles.inds_parallel_to
    @skip_rewrite_func Tangles.size_inds
    @skip_rewrite_func Tangles.size_ind
    @skip_rewrite_func Tangles.tensor_at
    @skip_rewrite_func Tangles.ind_at
    @skip_rewrite_func Tangles.addtensor!
    @skip_rewrite_func Tangles.rmtensor!
    @skip_rewrite_func Tangles.replace_tensor!
    @skip_rewrite_func Tangles.replace_ind!

    # `Lattice` interface
    @skip_rewrite_type Tangles.sites
    @skip_rewrite_func Tangles.bonds
    @skip_rewrite_func Tangles.all_sites
    @skip_rewrite_func Tangles.all_bonds
    @skip_rewrite_func Tangles.all_sites_iter
    @skip_rewrite_func Tangles.all_bonds_iter
    @skip_rewrite_func Tangles.hassite
    @skip_rewrite_func Tangles.hasbond
    @skip_rewrite_func Tangles.nsites
    @skip_rewrite_func Tangles.nbonds
    @skip_rewrite_func Tangles.site_at
    @skip_rewrite_func Tangles.bond_at
    @skip_rewrite_func Tangles.setsite!
    @skip_rewrite_func Tangles.setbond!
    @skip_rewrite_func Tangles.unsetsite!
    @skip_rewrite_func Tangles.unsetbond!

    # `Pluggable` interface
    @skip_rewrite_func Tangles.plugs
    @skip_rewrite_func Tangles.plug
    @skip_rewrite_func Tangles.all_plugs
    @skip_rewrite_func Tangles.all_plugs_iter
    @skip_rewrite_func Tangles.hasplug
    @skip_rewrite_func Tangles.nplugs
    @skip_rewrite_func Tangles.plug_at
    @skip_rewrite_func Tangles.plugs_like
    @skip_rewrite_func Tangles.plug_like
    @skip_rewrite_func Tangles.plugs_set
    @skip_rewrite_func Tangles.setplug!
    @skip_rewrite_func Tangles.unsetplug!

    # `AbstractTensorNetwork` methods
    @skip_rewrite_func Tangles.contract
    @skip_rewrite_func Tangles.replace_inds!
    @skip_rewrite_func Tangles.resetinds!
    @skip_rewrite_func Tangles.adjoint_plugs!
    @skip_rewrite_func Tangles.align!

    # constructors
    @skip_rewrite_type Tangles.SimpleTensorNetwork
    @skip_rewrite_func Tangles.GenericTensorNetwork
end

function __init__()
    tangles_skip_rewrites()
end

# if Reactant.Reactant_jll.is_available() && Reactant.precompilation_supported()
#     @setup_workload begin
#         using Muscle
#         using Reactant

#         tangles_skip_rewrites()

#         # Initialize the MLIR dialects and set up the XLA client
#         # NOTE taken from https://github.com/EnzymeAD/Reactant.jl/blob/77a9c694c4004cf08b270d08f8a5f51b7bdbf97e/src/Precompile.jl#L57-L83
#         Reactant.initialize_dialect()
#         if Reactant.XLA.REACTANT_XLA_RUNTIME == "PJRT"
#             client = Reactant.XLA.PJRT.CPUClient(; checkcount=false)
#         elseif Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
#             client = Reactant.XLA.IFRT.CPUClient(; checkcount=false)
#         else
#             error("Unsupported runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)")
#         end

#         a = Tensor(Reactant.to_rarray(ones(2, 2); client), [:i, :j])
#         b = Tensor(Reactant.to_rarray(ones(2, 2); client), [:j, :k])
#         c = Tensor(Reactant.to_rarray(ones(2, 2, 2); client), [:k, :l, :i])

#         @compile_workload begin
#             tn = GenericTensorNetwork([a, b, c])
#             setsite!(tn, c, site"1")
#             setplug!(tn, Index(:l), plug"1")
#             Reactant.compile(Tangles.contract, (tn,); client, optimize=:all)
#         end

#         # clean deinitialization
#         Reactant.XLA.free_client(client)
#         client.client = C_NULL
#         Reactant.deinitialize_dialect()
#         Reactant.clear_oc_cache()
#     end
# end

end
