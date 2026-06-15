module TanglesAdaptExt

using Tangles
using Adapt

Adapt.adapt_structure(to, x::NamedTensor) = Tensor(adapt(to, parent(x)), inds(x))

function Adapt.adapt_structure(to, x::Tangles.AbstractTensorNetwork)
    y = copy(x)

    for tensor in all_tensors(y)
        replace_tensor!(y, tensor, adapt(to, tensor))
    end

    return y
end

end
