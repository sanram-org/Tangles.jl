using Muscle

function simple_update! end

# auxiliar functions
function acting_sites(operator::NamedTensor)
    @assert all(i -> isplug(i.label), inds(operator)) "Operator indices must be plugs to be treated as an operator"

    target_plugs = plugs(operator)
    target_plugs_dual = filter(isdual, target_plugs)
    target_plugs_normal = filter(!isdual, target_plugs)

    @assert issetequal(target_plugs_normal, adjoint.(target_plugs_dual)) "Operator must have same input and output plugs"

    return unique(site.(target_plugs_dual))
end

function generic_simple_update!(tn, operator; maxdim=nothing)
    op_sites = acting_sites(operator)
    @assert 1 <= length(op_sites) <= 2 "Operator must act on one or two sites"
    @assert all(Base.Fix1(hasplug, tn), Plug.(op_sites; isdual=false)) "Operator plugs must be present in the MPS"

    # TODO move to function?
    # TODO do not move orthogonality center if 1-site... and unitary?
    # shortcut for 1-site operator
    if length(op_sites) == 1
        _site = only(op_sites)
        _tensor = tensor_at(tn, _site)
        tmp_contracting_ind = Index(gensym(:tmp))
        tensor = replace(_tensor, ind_at(tn, plug"$_site") => tmp_contracting_ind)
        operator = replace(operator, Index(plug"$_site'") => tmp_contracting_ind)
        new_tensor = einsum(tensor, operator)
        replace_tensor!(tn, _tensor, new_tensor)
        return tn
    end

    site_a, site_b = minmax(op_sites...)
    old_tensor_a = tensor_at(tn, site_a)
    old_tensor_b = tensor_at(tn, site_b)

    tmp_contracting_ind_a = Index(gensym(:tmp))
    tmp_contracting_ind_b = Index(gensym(:tmp))

    tensor_a = replace(old_tensor_a, ind_at(tn, plug"$site_a") => tmp_contracting_ind_a)
    tensor_b = replace(old_tensor_b, ind_at(tn, plug"$site_b") => tmp_contracting_ind_b)

    operator = replace(
        operator, Index(plug"$site_a'") => tmp_contracting_ind_a, Index(plug"$site_b'") => tmp_contracting_ind_b
    )

    new_tensor_a, new_tensor_b = simple_update(
        tensor_a,
        tensor_b,
        operator;
        physical_inds=(tmp_contracting_ind_a, tmp_contracting_ind_b),
        bond_ind=ind_at(tn, bond"$site_a-$site_b"),
        maxdim,
        absorb=Muscle.AbsorbEqually(),
    )

    new_tensor_a = replace(new_tensor_a, tmp_contracting_ind_a => ind_at(tn, plug"$site_a"))
    new_tensor_b = replace(new_tensor_b, tmp_contracting_ind_b => ind_at(tn, plug"$site_b"))

    @unsafe_region tn begin
        replace_tensor!(tn, old_tensor_a, new_tensor_a)
        replace_tensor!(tn, old_tensor_b, new_tensor_b)
    end

    return tn
end

simple_update!(tn::AbstractTensorNetwork, operator::NamedTensor; kwargs...) = generic_simple_update!(tn, operator; kwargs...)

## `MPS`
function simple_update!(tn::MPS, operator::NamedTensor; kwargs...)
    op_sites = acting_sites(operator)

    # move orthogonality center to operator sites
    canonize!(tn, MixedCanonical(op_sites))

    # perform the simple update routine
    generic_simple_update!(tn, operator; kwargs...)

    return tn
end

## TODO `VidalMPS`
# function simple_update!(tn::VidalMPS, operator::NamedTensor; kwargs...)

#     # TODO
#     Λc = ...
#     Λl = ...
#     Λr = ...
#     Γl = ...
#     Γr = ...

#     # prepare orthogonality center around target sites
#     a = hadamard(hadamard(Γl, Λl), Λc)
#     b = hadamard(Γr, Λr)

#     # perform simple update routine
#     new_a, new_Λc, new_b = Muscle.simple_update(a, b, ...; absorb=Muscle.DontAbsorb())

#     # recover gamma, lambda from updated tensors
# end
