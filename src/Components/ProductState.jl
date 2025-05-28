using Networks
using TenetCore
using QuantumTags
using LinearAlgebra

abstract type AbstractProduct <: AbstractTangle end

"""
    ProductState

A Tensor Network representing a product state.
"""
struct ProductState <: AbstractProduct
    tn::GenericTensorNetwork
end

"""
    ProductOperator

A Tensor Network representing a product operator.
"""
struct ProductOperator <: AbstractProduct
    tn::GenericTensorNetwork
end

@deprecate Product(arrays::AbstractArray{<:AbstractVector}) ProductState(arrays)
@deprecate Product(arrays::AbstractArray{<:AbstractMatrix}) ProductOperator(arrays)

ImplementorTrait(interface, tn::AbstractProduct) = ImplementorTrait(interface, tn.tn)
function DelegatorTrait(interface, tn::AbstractProduct)
    if ImplementorTrait(interface, tn.tn) === Implements()
        DelegateTo{:tn}()
    else
        DontDelegate()
    end
end

Base.copy(tn::P) where {P<:AbstractProduct} = P(copy(tn.tn))
Base.similar(tn::P) where {P<:AbstractProduct} = P(similar(tn.tn))
Base.zero(tn::P) where {P<:AbstractProduct} = P(zero(tn.tn))

# constructors
function ProductState(arrays::AbstractArray{<:AbstractVector})
    tn = GenericTensorNetwork()
    for coord in eachindex(IndexCartesian(), arrays)
        _tensor = Tensor(arrays[coord], [Index(plug"coord")])
        addtensor!(tn, _tensor)
        tag!(tn, _tensor, CartesianSite(coord))
        tag!(tn, Index(plug"coord"), plug"coord")
    end

    return ProductState(tn)
end

function ProductState(tensors::AbstractArray{<:Tensor})
    @assert all(==(1) ∘ ndims, tensors)

    tn = GenericTensorNetwork()
    for coord in eachindex(IndexCartesian(), tensors)
        _tensor = tensors[coord]
        addtensor!(tn, _tensor)
        index = only(inds(_tensor))
        tag!(tn, index, plug"coord")
    end

    return tn
end

function ProductState(bitstr::String)
    arrays = map(bitstr) do bitchar
        if bitchar === '0'
            [1.0, 0.0]
        elseif bitchar === '1'
            [0.0, 1.0]
        elseif bitchar === '+'
            [1.0, 1.0] / sqrt(2)
        elseif bitchar === '-'
            [1.0, -1.0] / sqrt(2)
        elseif bitchar === '↑' || bitchar === 'u'
            [1.0, -1.0im] / sqrt(2)
        elseif bitchar === '↓' || bitchar === 'd'
            [-1.0, -1.0im] / sqrt(2)
        else
            throw(ArgumentError("invalid character: $bitchar"))
        end
    end

    return ProductState(arrays)
end

function ProductOperator(arrays::AbstractArray{<:AbstractMatrix})
    tn = GenericTensorNetwork()

    for coord in eachindex(IndexCartesian(), arrays)
        _tensor = Tensor(arrays[coord], [Index(plug"coord"), Index(plug"coord'")])
        addtensor!(tn, _tensor)
        tag!(tn, _tensor, CartesianSite(coord))
        tag!(tn, Index(plug"coord"), plug"coord")
        tag!(tn, Index(plug"coord'"), plug"coord'")
    end

    return ProductOperator(tn)
end

function ProductOperator(tensors::AbstractArray{<:Tensor})
    @assert all(==(2) ∘ ndims, tensors)

    tn = GenericTensorNetwork()
    for coord in eachindex(IndexCartesian(), tensors)
        tensor = tensors[coord]
        addtensor!(tn, tensor)
        ind_out, ind_in = inds(tensor)
        tag!(tn, ind_out, plug"coord")
        tag!(tn, ind_in, plug"coord'")
    end

    return tn
end

# derived methods
function LinearAlgebra.norm(tn::AbstractProduct, p::Real=2)
    mapreduce(*, tensors(tn)) do tensor
        norm(parent(tensor), p) # TODO is this implemented?
    end
end

function LinearAlgebra.opnorm(tn::ProductOperator; p::Real=2)
    return mapreduce(*, tensors(tn)) do tensor
        opnorm(parent(tensor), p)
    end
end

LinearAlgebra.normalize(tn::AbstractProduct; kwargs...) = normalize!(copy(tn); kwargs...)

function LinearAlgebra.normalize!(tn::AbstractProduct; p::Real=2)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    return tn
end

# TODO move to some interface package? to TenetCore?
function overlap(a::ProductState, b::ProductState)
    issetequal(sites(a), sites(b)) || throw(ArgumentError("Both `ProductStates` must have the same sites"))
    return mapreduce(*, sites(a)) do site
        dot(tensor(a; at=site), conj(tensor(b; at=site)))
    end
end
