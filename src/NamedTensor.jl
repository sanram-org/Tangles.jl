using Base: @propagate_inbounds, AbstractVecOrTuple
using Base.Broadcast: Broadcasted, ArrayStyle
using LinearAlgebra
using Muscle
import Muscle: Tensor, variance, platform, extend, expand, fuse, isisometry, einsum, einsum!, factordims, tensor_qr, tensor_svd, tensor_eigen, simple_update

"""
    NamedTensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}

An array-like object with named dimensions (i.e. [`Index`](@ref)).
"""
struct NamedTensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::Tensor{T,N,A}
    inds::IndexList

    function NamedTensor(data::Tensor{T,N,A}, inds::IndexList) where {T,N,A<:AbstractArray{T,N}}
        _nonunique_inds = nonunique(inds)
        if !isempty(_nonunique_inds) &&
            !all(i -> allequal(Iterators.map(dim -> size(data, dim), findall(==(i), inds))), _nonunique_inds)
            throw(DimensionMismatch("nonuniform size of repeated indices"))
        end

        return new{T,N,A}(data, inds)
    end

    """
        NamedTensor(data::Tensor, inds)
    """
    function NamedTensor(data::Tensor, inds)
        return NamedTensor(data, IndexList(inds))
    end
end

NamedTensor(data::AbstractArray, inds) = NamedTensor(Tensor(data), inds)

NamedTensor(data::AbstractArray{T,0}) where {T} = NamedTensor(data, IndexList())
NamedTensor(data::Tensor{T,0}) where {T} = NamedTensor(data, IndexList())
NamedTensor(data::Number) = NamedTensor(fill(data))

NamedTensor(x::NamedTensor) = x
NamedTensor{T,N,A}(x::NamedTensor{T,N,A}) where {T,N,A} = x
function NamedTensor{T,N,A}(x::NamedTensor) where {T,N,A}
    throw(ArgumentError("NamedTensor type mismatch: $(typeof(x)) is not a NamedTensor{T,N,A}"))
end

NamedTensor(::NamedTensor, _) = throw(ArgumentError("Can't wrap a `NamedTensor` with another `NamedTensor`"))
function NamedTensor{T,N,A}(::NamedTensor, _) where {T,N,A}
    throw(ArgumentError("Can't wrap a `NamedTensor` with another `NamedTensor`"))
end

# useful shortcut
Tensor(data::AbstractArray, inds::AbstractVecOrTuple{Index}) = NamedTensor(data, inds)

"""
    Base.parent(::NamedTensor)

Return the underlying array of the tensor.
"""
Base.parent(t::NamedTensor) = t.data

"""
    inds(::NamedTensor)

Return the indices of the `NamedTensor`.
"""
inds(x::NamedTensor) = x.inds

"""
    dim(tensor::NamedTensor, i)

Return the location of the dimension of `tensor` corresponding to the given index `i`.
"""
dim(::NamedTensor, i::Number) = i
dim(t::NamedTensor, i::Symbol) = dim(t, Index(i))
dim(t::NamedTensor, i::Index) = findfirst(==(i), inds(t))

variance(x::NamedTensor) = variance(x.data)
variance(x::NamedTensor, i) = variance(x.data, dim(x, i))

platform(x::NamedTensor) = platform(parent(x))

arraytype(::Type{NamedTensor{T,N,A}}) where {T,N,A} = A
arraytype(::T) where {T<:NamedTensor} = arraytype(T)

Base.copy(t::NamedTensor{T,N,<:SubArray{T,N}}) where {T,N} = NamedTensor(copy(parent(t)), copy(inds(t)))

Base.print_array(io::IO, tensor::NamedTensor) = Base.print_array(io, parent(tensor))
function Base.showarg(io::IO, tensor::NamedTensor, toplevel)
    toplevel || print(io, "::")
    print(io, "NamedTensor(")
    Base.showarg(io, parent(tensor), false)
    print(io, ")")
    ndims(tensor) > 0 && print(io, " with signature $(Muscle.index_signature(parent(tensor)))")
    return nothing
end

"""
    Base.similar(::NamedTensor{T,N}[, S::Type, dims::Base.Dims{N}; inds])

Return a uninitialize tensor of the same size, eltype and [`inds`](@ref) as `tensor`. If `S` is provided, the eltype of the tensor will be `S`. If `dims` is provided, the size of the tensor will be `dims`.
"""
Base.similar(t::NamedTensor; inds=inds(t)) = NamedTensor(similar(parent(t)), inds)
Base.similar(t::NamedTensor, S::Type; inds=inds(t)) = NamedTensor(similar(parent(t), S), inds)
function Base.similar(t::NamedTensor{T,N}, S::Type, dims::Base.Dims{N}; inds=inds(t)) where {T,N}
    return NamedTensor(similar(parent(t), S, dims), inds)
end
function Base.similar(t::NamedTensor, ::Type, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end
Base.similar(t::NamedTensor{T,N}, dims::Base.Dims{N}; inds=inds(t)) where {T,N} = NamedTensor(similar(parent(t), dims), inds)
function Base.similar(t::NamedTensor, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end

"""
    Base.zero(tensor::NamedTensor)

Return a tensor of the same size, eltype and [`inds`](@ref) as `tensor` but filled with zeros.
"""
Base.zero(t::NamedTensor) = NamedTensor(zero(parent(t)), inds(t))

Base.:(==)(a::AbstractArray, b::NamedTensor) = isequal(b, a)
Base.:(==)(a::NamedTensor, b::AbstractArray) = isequal(a, b)
Base.:(==)(a::NamedTensor, b::NamedTensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::NamedTensor) = false
Base.isequal(a::NamedTensor, b::AbstractArray) = false
function Base.isequal(a::NamedTensor, b::NamedTensor)
    issetequal(inds(a), inds(b)) || return false
    perm = findperm(inds(a), inds(b))
    return isequal(parent(a), permutedims(parent(b), perm))
end

Base.isequal(a::NamedTensor{A,0}, b::NamedTensor{B,0}) where {A,B} = isequal(only(a), only(b))

Base.isapprox(a::AbstractArray, b::NamedTensor) = false
Base.isapprox(a::NamedTensor, b::AbstractArray) = false
function Base.isapprox(a::NamedTensor, b::NamedTensor; kwargs...)
    issetequal(inds(a), inds(b)) || return false
    perm = findperm(inds(a), inds(b))
    return isapprox(parent(a), permutedims(parent(b), perm); kwargs...)
end

Base.isapprox(a::NamedTensor{T,0}, b::T; kwargs...) where {T} = isapprox(only(a), b; kwargs...)
Base.isapprox(a::T, b::NamedTensor{T,0}; kwargs...) where {T} = isapprox(b, a; kwargs...)
Base.isapprox(a::NamedTensor{A,0}, b::NamedTensor{B,0}; kwargs...) where {A,B} = isapprox(only(a), only(b); kwargs...)

# NOTE: `replace` does not currenly support cyclic replacements
"""
    Base.replace(::NamedTensor, old_new::Pair{Index,Index}...)

Replace the indices of the tensor according to the given pairs of old and new indices.

!!! warning

    This method does not support cyclic replacements.
"""
Base.replace(t::NamedTensor, old_new::Pair...) = NamedTensor(parent(t), replace(inds(t), old_new...))

# Iteration interface
Base.IteratorSize(::Type{NamedTensor{T,N,A}}) where {T,N,A} = Base.IteratorSize(A)
Base.IteratorEltype(::Type{NamedTensor{T,N,A}}) where {T,N,A} = Base.IteratorEltype(A)

# Indexing interface
Base.IndexStyle(::Type{NamedTensor{T,N,A}}) where {T,N,A} = IndexStyle(A)

"""
    Base.getindex(::NamedTensor, i...)
    Base.getindex(::NamedTensor; i...)
    (::NamedTensor)[index=i...]

Return the element of the tensor at the given indices.
"""
@propagate_inbounds Base.getindex(t::NamedTensor, i...) = getindex(parent(t), i...)

# `tensor[Index(...) => 1]` case
@propagate_inbounds function Base.getindex(t::NamedTensor, i::Pair...)
    extent = _getindex_canonical_keys(t, i)
    return getindex(parent(t), extent...)
end

# `tensor[]` case and `tensor[i=1]` case where `Index(:i)` is in `inds(t)`
@propagate_inbounds function Base.getindex(t::NamedTensor; i...)
    length(i) == 0 && return getindex(parent(t))
    return getindex(t, i...)
end

_inds_getindex_nonsingleton(t::NamedTensor, i) = inds(t)[_view_singleton_mask(Val(ndims(t)), i)]

function _view_singleton_mask(::Val{N}, i) where {N}
    mask = falses(N)
    for (idx, ii) in enumerate(i)
        mask[idx] = ii isa Integer ? false : true
    end
    return mask
end

function _getindex_canonical_keys(t::NamedTensor, kv)
    _inds = Any[]
    sizehint!(_inds, ndims(t))
    for ind in inds(t)
        i = findfirst(x -> Index(x) == ind, Iterators.map(first, kv))
        push!(_inds, !isnothing(i) ? kv[i].second : Colon())
    end
    return _inds
end

"""
    Base.setindex!(t::NamedTensor, v, i...)
    Base.setindex(::NamedTensor; i...)
    (::NamedTensor)[index=i...]

Set the element of the tensor at the given indices to `v`.
"""
@propagate_inbounds Base.setindex!(t::NamedTensor, v, i...) = setindex!(parent(t), v, i...)
@propagate_inbounds function Base.setindex!(t::NamedTensor, v, i::Pair...)
    extent = _getindex_canonical_keys(t, i)
    setindex!(parent(t), v, extent...)
    return t
end

@propagate_inbounds function Base.setindex!(t::NamedTensor, v; i...)
    length(i) == 0 && return setindex!(parent(t), v)
    return setindex!(t, v, i...)
end

Base.firstindex(t::NamedTensor) = firstindex(parent(t))
Base.lastindex(t::NamedTensor) = lastindex(parent(t))

# AbstractArray interface
Base.eltype(x::NamedTensor) = eltype(x.data)

"""
    Base.size(::NamedTensor[, i::Index])

Return the size of the underlying array. If the dimension `i` (specified by `Index` or `Integer`) is specified, then the size of the corresponding dimension is returned.
"""
Base.size(t::NamedTensor) = size(parent(t))
Base.size(t::NamedTensor, i) = size(parent(t), dim(t, i))

"""
    Base.length(::NamedTensor)

Return the length of the underlying array.
"""
Base.length(t::NamedTensor) = length(parent(t))

Base.axes(t::NamedTensor) = axes(parent(t))
Base.axes(t::NamedTensor, d) = axes(parent(t), dim(t, d))

# StridedArrays interface
Base.strides(t::NamedTensor) = strides(parent(t))
Base.stride(t::NamedTensor, i) = stride(parent(t), dim(t, i))
# fix ambiguity
Base.stride(t::NamedTensor, i::Integer) = stride(parent(t), i)
Base.unsafe_convert(::Type{Ptr{T}}, t::NamedTensor{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(t))
Base.elsize(::Type{NamedTensor{T,N,A}}) where {T,N,A} = Base.elsize(A)

# Broadcasting
Base.BroadcastStyle(::Type{T}) where {T<:NamedTensor} = ArrayStyle{T}()

function Base.similar(bc::Broadcasted{ArrayStyle{NamedTensor{T,N,A}}}, ::Type{ElType}) where {T,N,A,ElType}
    # NOTE already checked if dimension mismatch
    # TODO throw on label mismatch?
    tensor = first(arg for arg in bc.args if arg isa NamedTensor{T,N,A})
    return similar(tensor, ElType)
end

"""
    Base.selectdim(tensor::NamedTensor, dim::Index, i)
    Base.selectdim(tensor::NamedTensor, dim::Integer, i)

Return a view of the tensor where the index for dimension `dim` equals `i`.

!!! note

    This method doesn't return a `SubArray`, but a `NamedTensor` wrapping a `SubArray`.

See also: [`selectdim`](@ref)
"""
Base.selectdim(t::NamedTensor, d::Integer, i) = NamedTensor(selectdim(parent(t), d, i), inds(t))

function Base.selectdim(t::NamedTensor{T,N}, d::Integer, i::Integer) where {T,N}
    data = selectdim(parent(t), d, i)
    indices = Index[label for (i, label) in enumerate(inds(t)) if i != d]
    return NamedTensor(data, indices)
end

Base.selectdim(t::NamedTensor, d, i) = selectdim(t, dim(t, d), i)

"""
    Base.permutedims(tensor::NamedTensor, perm)

Permute the dimensions of `tensor` according to the given permutation `perm`. The [`inds`](@ref) will be permuted accordingly.
"""
function Base.permutedims(t::NamedTensor, perm)
    _inds = Index[]
    for i in perm
        push!(_inds, inds(t)[i])
    end
    NamedTensor(permutedims(parent(t), perm), _inds)
end

# shortcut for 0-dimensional tensors
Base.permutedims(t::NamedTensor{T,0}, _) where {T} = t
Base.permutedims(t::NamedTensor{T,0}, ::Base.AbstractVecOrTuple{Index}) where {T} = t

Base.permutedims!(dest::NamedTensor, src::NamedTensor, perm) = permutedims!(parent(dest), parent(src), perm)

function Base.permutedims(t::NamedTensor{T}, perm::Base.AbstractVecOrTuple{Index}) where {T}
    perm = Int[findfirst(is_equal_label(ind), inds(t)) for ind in perm]
    return permutedims(t, perm)
end

"""
    Base.dropdims(tensor::NamedTensor; dims)

Return a tensor where the dimensions specified by `dims` are removed. `size(tensor, dim) == 1` for each dimension in `dims`.
"""
function Base.dropdims(t::NamedTensor; dims=tuple(findall(==(1), size(t))...))
    return NamedTensor(dropdims(parent(t); dims), inds(t)[setdiff(1:ndims(t), dims)])
end

"""
    Base.view(tensor::NamedTensor, i...)
    Base.view(tensor::NamedTensor, inds::Pair{Index,<:Any}...)

Return a view of the tensor with the given indices. If a `Pair` is given, the index is replaced by the value of the pair.

!!! note

    This method doesn't return a `SubArray`, but a `NamedTensor` wrapping a `SubArray`.
"""
function Base.view(t::NamedTensor, i...)
    return NamedTensor(view(parent(t), i...), _inds_getindex_nonsingleton(t, i))
end

# `@view tensor[Index(...) => 1]` case
function Base.view(t::NamedTensor, kv::Pair...)
    extent = _getindex_canonical_keys(t, kv)
    data = view(parent(t), extent...)
    _inds = _inds_getindex_nonsingleton(t, extent)
    return NamedTensor(data, _inds)
end

# `@view tensor[]` case and `@view tensor[i=1]` case where `Index(:i)` is in `inds(t)`
function Base.view(t::NamedTensor; kw...)
    length(kw) == 0 && return NamedTensor(view(parent(t)))
    return view(t, kw...)
end

# NOTE: `conj` is automatically managed because `NamedTensor` inherits from `AbstractArray`,
# but there is a bug when calling `conj` on `NamedTensor{T,0}` which makes it return a `NamedTensor{NamedTensor{Complex, 0}, 0}`
"""
    Base.conj(::NamedTensor)

Return the conjugate of the tensor.
"""
Base.conj(x::NamedTensor{<:Complex,0}) = NamedTensor(conj(parent(x)))

"""
    Base.adjoint(::NamedTensor)

Return the adjoint of the tensor.
"""
Base.adjoint(t::NamedTensor) = NamedTensor(adjoint(parent(t)), copy(inds(t)))

# NOTE: Maybe use transpose for lazy transposition ?
Base.transpose(t::NamedTensor{T,1,A}) where {T,A<:AbstractArray{T,1}} = copy(t)
Base.transpose(t::NamedTensor{T,2,A}) where {T,A<:AbstractArray{T,2}} = NamedTensor(transpose(parent(t)), reverse(inds(t)))

"""
    extend(tensor::NamedTensor, ind::Index; [axis=1, size=1, method=:zeros, variance=Invariant])

Expand the tensor by adding a new [`Index`](@ref) `ind` with the given `size` at the specified `axis`.
"""
function extend(tensor::NamedTensor, ind::Index; axis=1, kwargs...)
    data = extend(parent(tensor); axis, kwargs...)
    indices = (inds(tensor)[1:(axis - 1)]..., label, inds(tensor)[axis:end]...)
    return NamedTensor(data, indices)
end

"""
    expand(tensor::NamedTensor, ind::Index, size[; method=:zeros])

Pad the tensor along the dimension specified by `ind` to reach new `size`.
Supported methods are `:zeros` and `:rand`.
"""
function expand(tensor::NamedTensor, ind::Index, _size; kwargs...)
    data = expand(parent(tensor), dim(tensor, ind), _size; kwargs...)
    return NamedTensor(data, copy(inds(tensor)))
end

Base.cat(tensor::NamedTensor) = tensor

"""
    Base.cat(a::NamedTensor, b::NamedTensor; dims)

Concatenate two tensors `a` and `b` along the specified dimensions `dims`.

The indices of the tensors must be equal, otherwise the second tensor will be permuted to match the first one.

!!! note

    `dims` must be a list of `Index`.
"""
function Base.cat(a::NamedTensor, b::NamedTensor; dims)
    dims = dims isa Index ? [dims] : dims
    @assert issetequal(inds(a), inds(b)) "Indices of tensors must be equal, got $(inds(a)) and $(inds(b))"
    @assert all(i -> size(a, i) == size(b, i), setdiff(inds(a), dims)) "Sizes of tensors must be equal in all dimensions except for the concatenation dimensions"

    if inds(a) != inds(b)
        b = permutedims(b, inds(a))
    end

    _dims = map(Base.Fix1(dim, a), dims)
    data = cat(parent(a), parent(b); dims=_dims)
    return NamedTensor(data, copy(inds(a)))
end

Base.cat(tensors::NamedTensor...; kwargs...) = foldl((a, b) -> cat(a, b; kwargs...), tensors)

LinearAlgebra.opnorm(x::NamedTensor, p::Real) = opnorm(parent(x), p)

# TODO choose a new index name? currently choosing the first index of `parinds`
"""
    fuse(tensor, parinds; ind=first(parinds))

Fuses `parinds`, leaves them on the right-side internally permuted with `permutator` and names it as `ind`.
"""
function fuse(tensor::NamedTensor, parinds, new_ind=first(parinds))
    @assert allunique(inds(tensor))
    @assert parinds ⊆ inds(tensor)

    dims = map(Base.Fix1(dim, tensor), parinds)
    data = fuse(parent(tensor), dims)

    newinds = [filter(∉(parinds), inds(tensor))..., new_ind]
    return NamedTensor(data, newinds)
end

function Base._mapreduce_dim(f, op, init, tensor::NamedTensor, ind::Index)
    Base._mapreduce_dim(f, op, init, parent(tensor), dim(tensor, ind))
end
function Base._mapreduce_dim(f, op, init, tensor::NamedTensor, c::Colon)
    Base._mapreduce_dim(f, op, init, parent(tensor), c)
end
function Base._mapreduce_dim(f, op, init, tensor::NamedTensor, dims)
    Base._mapreduce_dim(f, op, init, parent(tensor), dim.((tensor,), dims))
end

# fix for ambiguity
function Base._mapreduce_dim(f, op, init::Base._InitialValue, t::NamedTensor, c::Colon)
    Base._mapreduce_dim(f, op, init, parent(t), c)
end

Base._sum(x::NamedTensor, ind::Index; kwargs...) = NamedTensor(Base._sum(parent(x), dim(x, ind); kwargs...), inds(x))
Base._sum(x::NamedTensor, c::Colon; kwargs...) = NamedTensor(fill(Base._sum(parent(x), c; kwargs...)))
Base._sum(x::NamedTensor, dims; kwargs...) = NamedTensor(Base._sum(parent(x), dim.((x,), dims); kwargs...), inds(x))

function isisometry(tensor::NamedTensor, _inds; kwargs...)
    @assert _inds ⊆ inds(tensor) "Indices $_ind is not in the tensor indices $(inds(tensor))"
    dims = map(Base.Fix1(dim, tensor), _inds)
    return isisometry(parent(tensor), dims; kwargs...)
end

__einsum_inds_to_dims(_, _, dims::AbstractVecOrTuple{<:AbstractVecOrTuple{<:Integer}}) = dims
function __einsum_inds_to_dims(a, b, dims::AbstractVecOrTuple{Index})
    left = map(Base.Fix1(dim, a), dims)
    right = map(Base.Fix1(dim, b), dims)
    return left, right
end

function einsum(a::NamedTensor, b::NamedTensor)
    dims = inds(a) ∩ inds(b)
    left, right = __einsum_inds_to_dims(a, b, dims)
    data = einsum(parent(a), parent(b); dims=(left, right))
    _inds = Index[
        [inds(a)[d] for d in 1:ndims(a) if d ∉ left];
        [inds(b)[d] for d in 1:ndims(b) if d ∉ right]
    ]
    return NamedTensor(data, _inds)
end

function einsum!(c::NamedTensor, a::NamedTensor, b::NamedTensor)
    dims = inds(a) ∩ inds(b)
    @assert isdisjoint(dims, inds(c))
    # TODO check inds of `c` are ok
    left, right = __einsum_inds_to_dims(a, b, dims)
    einsum!(parent(c), parent(a), parent(b); dims=(left, right))
    return c
end

factordims(x::NamedTensor) = factordims(parent(x))
factordims(x::NamedTensor, left::Vector{<:Integer}) = factordims(parent(x), left)
factordims(x::NamedTensor, left::Base.AbstractVecOrTuple{Index}) = factordims(x, map(Base.Fix1(dim, x), left))

function factordims(x::NamedTensor, dims::Vector)
    @assert length(dims) == 2
    return factordims(x, dims[1], dims[2])
end

factordims(x::NamedTensor, dims::NTuple{2,Vector{Int}}) = factordims(parent(x), dims)
factordims(x::NamedTensor, dims::Vector{Vector{Int}}) = factordims(parent(x), dims)

function factordims(x::NamedTensor, left::AbstractVecOrTuple{Int}, right::AbstractVecOrTuple{Int})
    return factordims(parent(x), (left, right))
end

function factordims(x::NamedTensor, left::AbstractVecOrTuple{Index}, right::AbstractVecOrTuple{Index})
    return factordims(x, map(Base.Fix1(dim, x), left), map(Base.Fix1(dim, x), right))
end

function factorinds(x::NamedTensor, args...)
    left, right = factordims(x, args...)
    return Index[inds(x)[i] for i in left], Index[inds(x)[i] for i in right]
end

function factorinds(x::NamedTensor, _left::Base.AbstractVecOrTuple{Index})
    left, right = factordims(x, map(Base.Fix1(dim, x), _left))
    return Index[inds(x)[i] for i in left], Index[inds(x)[i] for i in right]
end

function factorinds(x::NamedTensor, dims::Base.AbstractVecOrTuple{<:Base.AbstractVecOrTuple{Index}})
    @assert length(dims) == 2
    @assert isdisjoint(dims[1], dims[2])
    @assert dims[1] ⊆ inds(x)
    @assert dims[2] ⊆ inds(x)
    return dims
end

"""
    tensor_qr(A::NamedTensor; vind::Index, dims=factordims(A), kwargs...)

Perform QR factorization on a tensor. `dims` should be of the form `(dims_q, dims_r)` or just `(dims_q...,)`.
If `dims` is not set, then [`Covariant`](@ref) and [`Contravariant`] dimensions will be used as left- and right-dimensions.
"""
function tensor_qr(a::NamedTensor; vind::Index, dims=factordims(a), kwargs...)
    _dims = factordims(a, dims)
    linds, rinds = factorinds(a, dims)
    @show _dims linds rinds
    data_q, data_r = tensor_qr(parent(a); dims=_dims, kwargs...)
    q = NamedTensor(data_q, Index[linds; vind])
    r = NamedTensor(data_r, Index[vind; rinds])
    return q, r
end

LinearAlgebra.qr(a::NamedTensor; kwargs...) = tensor_qr(a; kwargs...)

"""
    tensor_svd(A::NamedTensor; vind::Index, dims, kwargs...)

Perform SVD factorization on a tensor.
If `dims` is not set, then [`Covariant`](@ref) and [`Contravariant`] dimensions will be used as left- and right-dimensions.
"""
function tensor_svd(a::NamedTensor; vind::Index, dims=factordims(a), kwargs...)
    _dims = factordims(a, dims)
    linds, rinds = factorinds(a, dims)
    data_u, data_s, data_vt = tensor_svd(parent(a); dims=_dims, kwargs...)
    u = NamedTensor(data_u, Index[linds; vind])
    s = NamedTensor(data_s, Index[vind])
    vt = NamedTensor(data_vt, Index[vind; rinds])
    return u, s, vt
end

LinearAlgebra.svd(a::NamedTensor; kwargs...) = tensor_svd(a; kwargs...)

"""
    tensor_eigen(tensor::NamedTensor; dims, kwargs...)

Perform eigen factorization on a tensor.
If `dims` is not set, then [`Covariant`](@ref) and [`Contravariant`] dimensions will be used as left- and right-dimensions.
"""
function tensor_eigen(a::NamedTensor; vind::Index, dims=factordims(a), kwargs...)
    _dims = factordims(a, dims)
    linds, rinds = factorinds(a, dims)
    data_λ, data_U = tensor_eigen(parent(a); dims=_dims, kwargs...)
    λ = NamedTensor(data_λ, Index[vind])
    U = NamedTensor(data_U, Index[linds; vind])
    return λ, U
end

LinearAlgebra.eigen(a::NamedTensor; kwargs...) = tensor_eigen(a; kwargs...)

function simple_update(
    a::NamedTensor,
    b::NamedTensor,
    g::NamedTensor;
    physical_inds = (inds(a)[findfirst(==(inds(g)[1]), inds(a))], inds(b)[findfirst(==(inds(g)[2]), inds(b))]),
    bond_ind = first(inds(a) ∩ inds(b)),
    kwargs...
)
    @assert bond_ind ∈ inds(a)
    @assert bond_ind ∈ inds(b)
    @assert physical_inds[1] ∈ inds(a)
    @assert physical_inds[2] ∈ inds(b)

    bond_dims = (dim(a, bond_ind), dim(b, bond_ind))
    physical_dims = (dim(a, physical_inds[1]), dim(b, physical_inds[2]))

    return simple_update(
        parent(a),
        parent(b),
        parent(g);
        physical_dims,
        bond_dims,
        kwargs...,
    )
end
