module Tangles

using Reexport

import EinExprs: inds

include("Utils.jl")

include("Tags.jl")
export @site_str, @bond_str, @plug_str, CartesianSite, Bond, Plug, Layer, InterLayer, LayerBond, InterLayerBond, LayerPlug

@reexport import Muscle: Tensor, Index

using DelegatorTraits
import DelegatorTraits: DelegatorTrait, ImplementorTrait, Implements, NotImplements

abstract type AbstractTensorNetwork end

# NOTE for developers
# try using functions owned by us (e.g. `mysize` instead of `Base.size`)
include("Interfaces/UnsafeScope.jl")
public UnsafeScopeable
export @unsafe_region

include("Interfaces/TensorNetwork.jl")
export TensorNetwork
export tensors,
    tensor, tensor_at, hastensor, ntensors, all_tensors, all_tensors_iter, addtensor!, rmtensor!, replace_tensor!
export inds, ind, ind_at, hasind, ninds, all_inds, all_inds_iter, replace_ind!
export tensors_with_inds, tensors_contain_inds, tensors_intersect_inds
export inds_set, inds_parallel_to
export size_inds, size_ind

include("Interfaces/Lattice.jl")
public Lattice
export sites,
    site, site_at, hassite, nsites, all_sites, all_sites_iter, sites_like, site_like, neighbor_sites, incident_sites
export bonds,
    bond, bond_at, hasbond, nbonds, all_bonds, all_bonds_iter, bonds_like, bond_like, neighbor_bonds, incident_bonds
export addsite!, addbond!, rmsite!, rmbond!, setsite!, setbond!, unsetsite!, unsetbond!

include("Interfaces/Pluggable.jl")
public Pluggable
export plugs,
    plug,
    plug_at,
    all_plugs,
    all_plugs_iter,
    nplugs,
    hasplug,
    plugs_like,
    plug_like,
    plugs_set_outputs,
    plugs_set_inputs,
    inds_set_physical,
    inds_set_virtual,
    inds_set_inputs,
    inds_set_outputs,
    setplug!,
    unsetplug!

# aliases to `Base` are in "src/Operations/AbstractTensorNetwork.jl"
include("Operations/TensorNetwork.jl")
export arrays, contract, resetinds!, canonicalize_inds!, cart_sites

include("Operations/Pluggable.jl")
export adjoint_plugs!, align!, @align!

include("Operations/AbstractTensorNetwork.jl")

# implementations
include("Components/GenericLattice.jl")
export GenericLattice

include("Components/SimpleTensorNetwork.jl")
export SimpleTensorNetwork

include("Components/GenericTensorNetwork.jl")
export GenericTensorNetwork

include("Components/LayeredTensorNetwork.jl")
export LayeredTensorNetwork

# extra
include("Operations/TensorExtra.jl")

# precompilation
using PrecompileTools

@setup_workload begin
    a = Tensor(ones(2, 2), [:i, :j])
    b = Tensor(ones(2, 2), [:j, :k])
    c = Tensor(ones(2, 2, 2), [:k, :l, :i])

    @compile_workload begin
        tn = GenericTensorNetwork([a, b, c])
        setsite!(tn, c, site"1")
        setplug!(tn, Index(:l), plug"1")
        Tangles.contract(tn)
    end
end

end
