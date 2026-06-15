module Tangles

using Reexport

import EinExprs: inds

include("Utils.jl")

include("Index.jl")
export Index

@reexport import Muscle:
    Tensor, variance, extend, expand, fuse, einsum, einsum!, tensor_qr, tensor_svd, tensor_eigen, simple_update
include("NamedTensor.jl")
export NamedTensor

using DelegatorTraits
import DelegatorTraits: DelegatorTrait, ImplementorTrait, Implements, NotImplements

abstract type AbstractTensorNetwork end

# NOTE for developers
# try using functions owned by us (e.g. `mysize` instead of `Base.size`)
include("Interfaces/UnsafeScope.jl")
public UnsafeScopeable
export @unsafe_region

include("Interfaces/TensorNetwork.jl")
public TensorNetwork
export tensors, tensor, hastensor, ntensors, all_tensors, all_tensors_iter, addtensor!, rmtensor!, replace_tensor!
export inds, ind, hasind, ninds, all_inds, all_inds_iter, replace_ind!
public tensors_set_equal, tensors_set_contain, tensors_set_intersect
public inds_set, inds_parallel_to
public size_inds, size_ind

# extra methods of `TensorNetwork`
export contract
public arrays, resetinds!, cart_sites

include("Tags.jl")
export @site_str,
    @bond_str, @plug_str, CartesianSite, Bond, Plug, Layer, InterLayer, LayerBond, InterLayerBond, LayerPlug

# helper method
plugs(tensor::NamedTensor) = filter!(isplug, map(x -> x.label, inds(tensor)))

include("Interfaces/TaggedTensorNetwork.jl")
public TaggedTensorNetwork
export sites, site, tensor_at, site_at, hassite, nsites, all_sites, all_sites_iter, neighbor_sites, site_incidents
export links, link, ind_at, link_at, haslink, nlinks, all_links, all_links_iter, neighbor_links, link_incidents
export setsite!, setlink!, unsetsite!, unsetlink!

# these methods are in the process of being reconsidered
export bonds, bond, all_bonds, all_bonds_iter, bond_at, hasbond, nbonds, neighbor_bonds
export plugs, plug, all_plugs, all_plugs_iter, plug_at, hasplug, nplugs
public plugs_set_in, plugs_set_out, plugs_set_dual

# extra methods of `TaggedTensorNetwork`
export @align!
public canonicalize_inds!, adjoint_plugs!, align!

# aliases to `Base` and other libraries
include("AbstractTensorNetwork.jl")

# implementations
include("Implementations/GenericLattice.jl")
export GenericLattice

include("Implementations/SimpleTensorNetwork.jl")
export SimpleTensorNetwork

include("Implementations/GenericTensorNetwork.jl")
export GenericTensorNetwork

include("Implementations/LayeredTensorNetwork.jl")
export LayeredTensorNetwork

# precompilation
using PrecompileTools

@setup_workload begin
    a = NamedTensor(ones(2, 2), Index.([:i, :j]))
    b = NamedTensor(ones(2, 2), Index.([:j, :k]))
    c = NamedTensor(ones(2, 2, 2), Index.([:k, :l, :i]))

    @compile_workload begin
        tn = GenericTensorNetwork([a, b, c])
        setsite!(tn, c, site"1")
        setlink!(tn, Index(:l), plug"1")
        Tangles.contract(tn)
    end
end

end
