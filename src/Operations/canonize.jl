using ArgCheck
using Muscle

"""
    CanonizeEffect{T}

The effect emitted by the `canonize!` function.
"""
struct CanonizeEffect{O,N} <: Effect
    old_form::O
    new_form::N
end

function canonize!(tn::Tangle, new_form; kwargs...)
    checkeffect(tn, CanonizeEffect(form(tn), new_form))
    canonize_inner!(tn, new_form; kwargs...)
    handle!(tn, CanonizeEffect(form(tn), new_form))
    return tn
end

canonize(tn, args...; kwargs...) = canonize!(copy(tn), args...; kwargs...)

function canonize_inner! end
