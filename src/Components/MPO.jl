using TenetCore

abstract type AbstractMPO <: Tangle end

defaultorder(::Type{<:AbstractMPO}) = (:o, :i, :l, :r)

"""
    MatrixProductOperator

A Matrix Product Operator (MPO) Tensor Network.
"""
mutable struct MatrixProductOperator <: AbstractMPO
    const tn::GenericTensorNetwork
    form::CanonicalFormTrait
end

const MPO = MatrixProductOperator
