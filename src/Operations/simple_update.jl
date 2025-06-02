using ArgCheck
using Muscle
using QuantumTags

function simple_update! end

# auxiliar functions
function generic_simple_update!(tn, operator; maxdim=nothing)
    @argcheck ndims(operator) == 4 "Operator must have 4 dimensions (2-site operator)"
    @argcheck all(isplug, inds(operator)) "Operator indices must be plugs to be treated as an operator"

    op_site = unique(site.(plugs(operator)))
    @assert length(op_site) == 2 "Operator must have exactly two sites"

    target_plugs = plugs(operator)
    target_plugs_dual = filter(isdual, target_plugs)
    target_plugs_normal = filter(!isdual, target_plugs)
    @argcheck issetequal(target_plugs_normal, adjoint.(target_plugs_dual)) "Operator must have same input and output plugs"

    @argcheck all(Base.Fix1(hasplug, tn), target_plugs_normal) "Operator plugs must be present in the MPS"

    site_a, site_b = minmax(site.(target_plugs_dual)...)
    old_tensor_a = tensor_at(tn, site_a)
    old_tensor_b = tensor_at(tn, site_b)

    tmp_contracting_ind_a = Index(gensym(:tmp))
    tmp_contracting_ind_b = Index(gensym(:tmp))

    tensor_a = replace(old_tensor_a, ind_at(tn, plug"site_a") => tmp_contracting_ind_a)
    tensor_b = replace(old_tensor_b, ind_at(tn, plug"site_b") => tmp_contracting_ind_b)

    operator = replace(
        operator, Index(plug"site_a'") => tmp_contracting_ind_a, Index(plug"site_b'") => tmp_contracting_ind_b
    )

    new_tensor_a, new_tensor_b = Muscle.simple_update(
        tensor_a,
        tmp_contracting_ind_a, # ind_physical_a,
        tensor_b,
        tmp_contracting_ind_b, # ind_physical_b,
        ind_at(tn, bond"site_a-site_b"), # ind_bond_ab,
        operator,
        Index(plug"site_a"), # ind_physical_op_a,
        Index(plug"site_b"); # ind_physical_op_b;
        maxdim,
        absorb=Muscle.AbsorbEqually(),
    )

    # fix the index renaming of `Muscle.simple_update`
    # TODO fix it better in Muscle?
    new_tensor_a = replace(new_tensor_a, tmp_contracting_ind_a => ind_at(tn, plug"site_a"))
    new_tensor_b = replace(new_tensor_b, tmp_contracting_ind_b => ind_at(tn, plug"site_b"))

    @unsafe_region tn begin
        replace_tensor!(tn, old_tensor_a, new_tensor_a)
        replace_tensor!(tn, old_tensor_b, new_tensor_b)
    end

    return tn
end

## `MPS`
simple_update!(tn::MPS, operator::Tensor; kwargs...) = generic_simple_update!(tn, operator; kwargs...)

## `MixedCanonicalMPS`
function simple_update!(tn::MixedCanonicalMPS, operator::Tensor; kwargs...)
    # move orthogonality center to operator sites
    canonize!(tn, MixedCanonical(op_site))

    # perform the simple update routine
    generic_simple_update!(tn, operator; kwargs...)

    return tn
end

## TODO `VidalMPS`
# function simple_update!(tn::VidalMPS, operator::Tensor; kwargs...)

#     # TODO
#     Λc = ...
#     Λl = ...
#     Λr = ...
#     Γl = ...
#     Γr = ...

#     # prepare orthogonality center around target sites
#     a = binary_einsum(binary_einsum(Λl, Γl), Λc)
#     b = binary_einsum(Γr, Λr)

#     # perform simple update routine
#     new_a, new_Λc, new_b = Muscle.simple_update(a, b, ...; absorb=Muscle.DontAbsorb())

#     # recover gamma, lambda from updated tensors
# end
