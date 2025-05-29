using LinearAlgebra: norm
import LinearAlgebra: normalize!

LinearAlgebra.normalize(tn::AbstractTangle; kwargs...) = normalize!(copy(tn); kwargs...)

# `AbstractProduct`
function LinearAlgebra.normalize!(tn::AbstractProduct; p::Real=2)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    return tn
end
