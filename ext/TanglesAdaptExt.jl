module TanglesAdaptExt

using Tangles
using Adapt

Adapt.adapt_structure(to, x::NamedTensor) = NamedTensor(adapt(to, parent(x)), inds(x))
Adapt.parent_type(::Type{NamedTensor{T,N,A}}) where {T,N,A} = Tensor{T,N,A}

function Adapt.adapt_structure(to, x::Tangles.AbstractTensorNetwork)
    y = copy(x)

    for tensor in all_tensors(y)
        replace_tensor!(y, tensor, adapt(to, tensor))
    end

    return y
end

end
