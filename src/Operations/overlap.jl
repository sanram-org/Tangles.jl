function overlap end

function overlap(a::ProductState, b::ProductState)
    issetequal(sites(a), sites(b)) || throw(ArgumentError("Both `ProductStates` must have the same sites"))
    return mapreduce(*, sites(a)) do site
        dot(tensor_at(a, site), conj(tensor_at(b, site)))
    end
end
