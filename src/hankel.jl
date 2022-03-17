const AbstractTensor{N,T} = AbstractArray{T,N}
const TupleN{T,N} = NTuple{N,T}

struct Hankel{T,N,C,K,A<:AbstractArray} <: AbstractArray{T,N}
    image::A
    channel_size::NTuple{C,Int}
    kernel_size::NTuple{K,Int}
    @doc raw"""
        Hankel(image, kernel_size, channel_size)

    Constructs a Hankel array `A` from the underlying `image` data, such that:

    ```julia
    A[c,j,k,b] = image[c,j+k-1,b]
    ```

    where `c,j,k,b` are multi-indices for the channel, kernel, image, and batch
    dimensions. So `j` is the kernel entry, and `k` the kernel position in the image.
    """
    function Hankel(image::AbstractArray, channel_size::NTuple{C,Int}, kernel_size::NTuple{K,Int}) where {C,K}
        input_size = size(image)[(C + 1):(C + K)]
        batch_size = size(image)[(C + K + 1):end]
        @assert size(image) == (channel_size..., input_size..., batch_size...)
        @assert all(input_size .≥ kernel_size)
        return new{eltype(image), ndims(image) + K, C, K, typeof(image)}(image, channel_size, kernel_size)
    end
end

function Hankel(image::AbstractArray, C::Int, kernel_size::NTuple{K,Int}) where {K}
    channel_size = size(v)[1:C]
    return Hankel(image, channel_size, kernel_size)
end

Base.IndexStyle(::Type{<:Hankel}) = IndexCartesian()
Base.size(A::Hankel) = (channel_size(A)..., kernel_size(A)..., output_size(A)..., batch_size(A)...)

image_size(A::Hankel) = size(A.image)
channel_size(A::Hankel) = A.channel_size
kernel_size(A::Hankel) = A.kernel_size
input_size(A::Hankel) = image_size(A)[(channel_ndims(A) + 1):(channel_ndims(A) + kernel_ndims(A))]
output_size(A::Hankel) = input_size(A) .- kernel_size(A) .+ 1
batch_size(A::Hankel) = image_size(A)[(channel_ndims(A) + kernel_ndims(A) + 1):end]

channel_ndims(A::Hankel) = length(channel_size(A))
kernel_ndims(A::Hankel) = length(kernel_size(A))

@inline function Base.getindex(A::Hankel{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        c = I[1:channel_ndims(A)]
        j = I[(channel_ndims(A) + 1):(channel_ndims(A) + kernel_ndims(A))]
        k = I[(channel_ndims(A) + kernel_ndims(A) + 1):(channel_ndims(A) + 2kernel_ndims(A))]
        b = I[(channel_ndims(A) + 2kernel_ndims(A) + 1):end]
        i = j .+ k .- 1
        return A.image[CartesianIndex(c), CartesianIndex(i), CartesianIndex(b)]
    end
end

@doc raw"""
    hankel(v, cdims, ksz)

Creates a Hankel array from `v` with channel dimensions `cdims` and kernel size `ksz`.
The returned array `A` satisfies:

```math
A_{i,j,k,n} = v_{i,j+k-1,n}
```

where `i,j,k,n` are multi-indices, with `i` traversing channels
and `j` traversing the kernel.
```
"""
function hankel(img::AbstractArray, channel_size::NTuple{C,Int}, kernel_size::NTuple{K,Int}) where {C,K}
    input_size = size(img)[(C + 1):(C + K)]
    batch_size = size(img)[(C + K + 1):end]
    @assert size(img) == (channel_size..., input_size..., batch_size...)
    output_size = input_size .- kernel_size .+ 1
    A = similar(img, eltype(img), channel_size..., kernel_size..., output_size..., batch_size...)
    vflat = reshape(img, prod(channel_size), input_size..., prod(batch_size))
    Aflat = reshape(A, prod(channel_size), kernel_size..., output_size..., prod(batch_size))
    flat_hankel!(Aflat, vflat)
    return A
end

function hankel(v::AbstractArray, C::Int, kernel_size::TupleN{Int})
    channel_size = size(v)[1:C]
    return hankel(v, channel_size, kernel_size)
end

# for 1 channel dimension and 1 batch dimension
flat_hankel!(A::AbstractTensor{4}, v::AbstractTensor{3}) =
    @tullio A[c,j1,k1,b] = v[c, j1 + k1 - 1, b]
flat_hankel!(A::AbstractTensor{6}, v::AbstractTensor{4}) =
    @tullio A[c,j1,j2,k1,k2,b] = v[c, j1 + k1 - 1, j2 + k2 - 1, b]
flat_hankel!(A::AbstractTensor{8}, v::AbstractTensor{5}) =
    @tullio A[c,j1,j2,j3,k1,k2,k3,b] = v[c, j1 + k1 - 1, j2 + k2 - 1, j3 + k3 - 1, b]
