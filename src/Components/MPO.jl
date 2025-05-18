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

ImplementorTrait(interface, tn::MPO) = ImplementorTrait(interface, tn.tn)
function DelegatorTrait(interface, tn::MPO)
    if ImplementorTrait(interface, tn.tn) === Implements()
        DelegateTo{:tn}()
    else
        DontDelegate()
    end
end

form(tn::MPO) = tn.form

Base.copy(tn::MPO) = MPO(copy(tn.tn), tn.form)

MPO(arrays; form::CanonicalFormTrait=NonCanonical(), kwargs...) = MPO(form, arrays; kwargs...)

function MPO(::NonCanonical, arrays::Vector; order=defaultorder(MPO))
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:(end - 1)]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"
    issetequal(order, defaultorder(MPO)) ||
        throw(ArgumentError("order must be a permutation of $(String.(defaultorder(MPO)))"))

    # n = length(arrays)
    # gen = IndexCounter()
    # lattice = Lattice(Val(:chain), n)

    # sitemap = Dict{Site,Symbol}(Site(i) => nextindex!(gen) for i in 1:n)
    # merge!(sitemap, Dict([Site(i; dual=true) => nextindex!(gen) for i in 1:n]))
    # bondmap = Dict{Bond,Symbol}(bond => nextindex!(gen) for bond in Graphs.edges(lattice))

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
            elseif dir == :i
                Index(plug"i'")
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
        tag!(tn, Index(plug"i'"), plug"i'")
        haslink(tn, bond"i-isup") || hasind(tn, Index(bond"i-isup")) && tag!(tn, Index(bond"i-isup"), bond"i-isup")
        haslink(tn, bond"isub-i") || hasind(tn, Index(bond"isub-i")) && tag!(tn, Index(bond"isub-i"), bond"isub-i")
    end

    return MPO(tn, NonCanonical())
end

function MPO(form::MixedCanonical, arrays::Vector; kwargs...)
    ψ = MPO(arrays; form=NonCanonical(), kwargs...)
    ψ.form = form
    return ψ
end
