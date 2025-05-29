using LinearAlgebra: norm
import LinearAlgebra: normalize, normalize!

normalize(tn::AbstractTangle; kwargs...) = normalize!(copy(tn); kwargs...)

# `AbstractProduct`
function normalize!(tn::AbstractProduct, p::Real=2)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    return tn
end

# `MixedCanonicalMPS`
# TODO what if `orthog_center` is not a single site?
normalize!(tn::MixedCanonicalMPS, p::Real=2) = normalize!(tensor_at(tn, tn.orthog_center.orthog_center), p)
