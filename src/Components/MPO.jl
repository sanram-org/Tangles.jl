using TenetCore

abstract type AbstractMPO <: AbstractTensorNetwork end

defaultorder(::Type{<:AbstractMPO}) = (:o, :i, :l, :r)

"""
    MPO <: AbstractMPO

A Matrix Product Operator (MPO) [`Ansatz`](@ref) Tensor Network.
"""
mutable struct MatrixProductOperator <: AbstractMPO
    const tn::GenericTensorNetwork
    form::CanonicalForm
end

const MPO = MatrixProductOperator
