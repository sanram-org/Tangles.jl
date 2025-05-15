using ArgCheck
using Muscle
import Muscle: simple_update
using QuantumTags

"""
    SimpleUpdateEffect{T}

The effect emitted by the `simple_update!` function.
"""
struct SimpleUpdateEffect{T} <: Effect
    op::T
end

# TODO can't write it more generically (i.e. needs `Tangle`) because `simple_update` belongs to `Muscle`
function simple_update!(tn::Tangle, op; kwargs...)
    checkeffect(tn, SimpleUpdateEffect(op))
    simple_update_inner!(tn, op; kwargs...)
    handle!(tn, SimpleUpdateEffect(op))
    return tn
end

simple_update(tn::Tangle, op; kwargs...) = simple_update!(copy(tn), op; kwargs...)

function checkeffect(tn, e::SimpleUpdateEffect)
    operator = e.op

    @argcheck ndims(operator) == 4 "Operator must have 4 dimensions (2-site operator)"
    @argcheck all(isplug, inds(operator)) "Operator indices must be plugs to be treated as an operator"

    target_plugs = plugs(operator)
    target_plugs_dual = filter(isdual, target_plugs)
    target_plugs_normal = filter(!isdual, target_plugs)

    @argcheck issetequal(target_plugs_normal, adjoint.(target_plugs_dual)) "Operator must have same input and output plugs"
    @argcheck all(Base.Fix1(hasplug, tn), target_plugs_normal) "Operator plugs must be present in the MPS"
end

handle!(_, ::SimpleUpdateEffect) = nothing

simple_update_inner!(tn::MPS, operator::Tensor; kwargs...) = simple_update_inner!(tn, operator, form(tn); kwargs...)

function simple_update_inner!(tn::MPS, operator, ::NonCanonical; maxdim=nothing)
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

    new_tensor_a, new_tensor_b = simple_update(
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

    replace_tensor!(tn, old_tensor_a, new_tensor_a)
    replace_tensor!(tn, old_tensor_b, new_tensor_b)

    return tn
end

function simple_update_inner!(tn::MPS, operator, orthog_form::MixedCanonical)
    # TODO canonize! to site
end

function simple_update_inner!(tn::MPS, operator, ::VidalGauge) end
