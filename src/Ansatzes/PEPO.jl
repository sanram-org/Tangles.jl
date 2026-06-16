using DelegatorTraits

struct ProjectedEntangledPairOperator <: AbstractTensorNetwork
    tn::GenericTensorNetwork
end

const PEPO = ProjectedEntangledPairOperator
Base.copy(tn::PEPO) = PEPO(copy(tn.tn))

defaultorder(::Type{PEPO}) = (:l, :r, :u, :d, :o, :i)

function PEPO(arrays::AbstractMatrix{<:AbstractArray}; order=defaultorder(PEPO))
    tn = GenericTensorNetwork()

    m, n = size(arrays)

    for I in eachindex(IndexCartesian(), arrays)
        (i, j) = Tuple(I)
        let dirs = collect(order)
            i == 1 && filter!(!=(:u), dirs)
            i == m && filter!(!=(:d), dirs)
            j == 1 && filter!(!=(:l), dirs)
            j == n && filter!(!=(:r), dirs)

            _inds = map(dirs) do dir
                if dir === :l
                    Index(bond"$(i, j) - $(i, j - 1)")
                elseif dir === :r
                    Index(bond"$(i, j) - $(i, j + 1)")
                elseif dir === :u
                    Index(bond"$(i, j) - $(i - 1, j)")
                elseif dir === :d
                    Index(bond"$(i, j) - $(i + 1, j)")
                elseif dir === :o
                    Index(plug"$i, $j")
                elseif dir === :i
                    Index(plug"$i, $j'")
                else
                    throw(ArgumentError("Invalid direction: $dir"))
                end
            end

            _tensor = NamedTensor(arrays[i, j], _inds)
            addtensor!(tn, _tensor)
            setsite!(tn, _tensor, site"$i,$j")

            _bonds = map(filter(x -> x !== :o, dirs)) do dir
                if dir === :l
                    bond"$(i, j) - $(i, j - 1)"
                elseif dir === :r
                    bond"$(i, j) - $(i, j + 1)"
                elseif dir === :u
                    bond"$(i, j) - $(i - 1, j)"
                elseif dir === :d
                    bond"$(i, j) - $(i + 1, j)"
                end
            end

            for _bond in _bonds
                if !haslink(tn, _bond)
                    setlink!(tn, Index(_bond), _bond)
                end
            end

            setlink!(tn, Index(plug"$i, $j"), plug"$i, $j")
            setlink!(tn, Index(plug"$i, $j'"), plug"$i, $j'")
        end
    end

    return PEPO(tn)
end

ImplementorTrait(interface, tn::PEPO) = ImplementorTrait(interface, tn.tn)
function DelegatorTrait(interface, tn::PEPO)
    if ImplementorTrait(interface, tn.tn) === Implements()
        DelegateToField{:tn}()
    else
        DontDelegate()
    end
end
