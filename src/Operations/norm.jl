import LinearAlgebra: norm

# `AbstractProduct`
function norm(tn::AbstractProduct, p::Real=2)
    mapreduce(*, tensors(tn)) do tensor
        norm(parent(tensor), p) # TODO is this implemented?
    end
end

# function LinearAlgebra.opnorm(tn::ProductOperator; p::Real=2)
#     return mapreduce(*, tensors(tn)) do tensor
#         opnorm(parent(tensor), p)
#     end
# end
