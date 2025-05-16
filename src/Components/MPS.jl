using TenetCore

abstract type AbstractMPS <: AbstractMPO end

defaultorder(::Type{<:AbstractMPS}) = (:o, :l, :r)

"""
    MatrixProductState

A Matrix Product State Tensor Network.
"""
mutable struct MatrixProductState <: AbstractMPS
    const tn::GenericTensorNetwork
    form::CanonicalFormTrait
end

const MPS = MatrixProductState

ImplementorTrait(interface, tn::MPS) = ImplementorTrait(interface, tn.tn)
function DelegatorTrait(interface, tn::MPS)
    if ImplementorTrait(interface, tn.tn) === Implements()
        DelegateTo{:tn}()
    else
        DontDelegate()
    end
end
form(tn::MPS) = tn.form

Base.copy(tn::MPS) = MPS(copy(tn.tn), tn.form)

MPS(arrays; form::CanonicalFormTrait=NonCanonical(), kwargs...) = MPS(form, arrays; kwargs...)
MPS(arrays::Vector{<:AbstractArray}, λ; kwargs...) = MPS(VidalGauge(), arrays, λ; kwargs...)

"""
    MPS(arrays::Vector{<:AbstractArray}; order=defaultorder(MPS))

Create a [`NonCanonical`](@ref) or [`MixedCanonical`](@ref) [`MPS`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:o, :l, :r)`.
"""
function MPS(::NonCanonical, arrays; order=defaultorder(MPS)) # , check=true)
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"
    issetequal(order, defaultorder(MPS)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPS)))"))

    tn = GenericTensorNetwork()

    for (i, array) in enumerate(arrays)
        isub = i - 1
        isup = i + 1

        local_order = if i == 1
            filter(x -> x != :l, order)
        elseif i == length(arrays)
            filter(x -> x != :r, order)
        else
            order
        end

        inds = map(local_order) do dir
            if dir == :o
                Index(plug"i")
            elseif dir == :r
                Index(bond"i-isup")
            elseif dir == :l
                Index(bond"isub-i")
            else
                throw(ArgumentError("Invalid direction: $dir"))
            end
        end |> collect

        _tensor = Tensor(array, inds)
        addtensor!(tn, _tensor)
        tag!(tn, _tensor, site"i")
        tag!(tn, Index(plug"i"), plug"i")
        haslink(tn, bond"i-isup") || hasind(tn, Index(bond"i-isup")) && tag!(tn, Index(bond"i-isup"), bond"i-isup")
        haslink(tn, bond"isub-i") || hasind(tn, Index(bond"isub-i")) && tag!(tn, Index(bond"isub-i"), bond"isub-i")
    end

    return MPS(tn, NonCanonical())
end

function MPS(form::MixedCanonical, arrays; order=defaultorder(MPS)) # , check=true)
    mps = MPS(arrays; form=NonCanonical(), order) #, check)
    mps.form = form
    # check && checkform(mps)
    return mps
end

"""
    MPS(VidalGauge(), Γ, λ; order=defaultorder(MPS), check=true)

Create a [`VidalGauge`](@ref) [`MPS`](@ref) from a vector of arrays.

# Keyword Arguments

  - `order` The order of the indices in the arrays. Defaults to `(:o, :l, :r)`.
  - `check` Whether to check the canonical form of the MPS.
"""
function MPS(::VidalGauge, Γ, λ; order=defaultorder(MPS)) # , check=true)
    @assert length(λ) == length(Γ) - 1 "Number of λ tensors must be one less than the number of Γ tensors"
    @assert all(==(1) ∘ ndims, λ) "All λ tensors must be 1-dimensional"

    tn = MPS(Γ; form=NonCanonical(), order, check)
    tn.form = VidalGauge()

    # create tensors from 'λ'
    map(enumerate(λ)) do (i, array)
        isup = i + 1
        bondind = ind(tn; at=bond"i-isup")
        _tensor = Tensor(array, [bondind])
        addtensor!(tn, _tensor)
    end

    # check canonical form by contracting Γ and λ tensors and checking their orthogonality
    # check && checkform(tn)

    return mps
end
