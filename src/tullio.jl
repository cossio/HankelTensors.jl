@doc raw"""
    conv_v2h(w, v)

Internal function used to compute inputs from a visible configurations `v` to the hidden
layer, where `w` are the convolutional RBM weights.

```math
I_\mu^{k_1,\dots,k_n,b} = \sum_{c,j_1,\dots,j_n} w_{c,j_1,\dots,j_n,\mu} v_{c,j_1+k_1-1,\dots,j_n+k_n-1,b}
```

Assumes that:

* `w` is of size `(C,J₁,...,Jₙ,M)`
* `v` is of size `(C,N₁,...,Nₙ,B)`

Here `C` is the flattened channel dimension,
`M` is the number of hidden units and `B` is the batch size.
Therefore the hidden and batch dimensions must be flattened before calling `conv_v2h`.
The output `I` is of size `(M, N₁ - J₁ + 1, ..., Nₙ - Jₙ + 1, B)`.

!!! warning
    This is an internal function and is not part of the public API.

!!! warning
    Only works for `n = 1, 2, 3` due to a technical limitations.
"""
function conv_v2h end

function conv_v2h(w::AbstractTensor{3}, v::AbstractTensor{3})
    _conv_v2h_check_size(w, v)
    @tullio I[μ,k,b] := w[c,j,μ] * v[c,j+k-1,b]
end

function conv_v2h(w::AbstractTensor{4}, v::AbstractTensor{4})
    _conv_v2h_check_size(w, v)
    @tullio I[μ,k₁,k₂,b] := w[c,j₁,j₂,μ] * v[c,j₁+k₁-1,j₂+k₂-1,b]
end

function conv_v2h(w::AbstractTensor{5}, v::AbstractTensor{5})
    _conv_v2h_check_size(w, v)
    @tullio I[μ,k₁,k₂,k₃,b] := w[c,j₁,j₂,j₃,μ] * v[c,j₁+k₁-1,j₂+k₂-1,j₃+k₃-1,b]
end

function conv_v2h(w::AbstractTensor{6}, v::AbstractTensor{6})
    _conv_v2h_check_size(w, v)
    @tullio I[μ,k₁,k₂,k₃,k₄,b] := w[c,j₁,j₂,j₃,j₄,μ] * v[c,j₁+k₁-1,j₂+k₂-1,j₃+k₃-1,j₄+k₄-1,b]
end

function _conv_v2h_check_size(w::AbstractTensor{N}, v::AbstractTensor{N}) where {N}
    @assert size(w, 1) == size(v, 1) # channel size
    @assert all((size(w) .≤ size(v))[2:(end - 1)])
end


@doc raw"""
    conv_h2v(w, v)

Internal function used to compute inputs from a hidden configurations `h` to the visible
layer, where `w` are the convolutional RBM weights.

```math
I_{i_1,\dots,i_n}^{k_1,\dots,k_n,b} = \sum_{c,j_1,\dots,j_n} w_{c,j_1,\dots,j_n,\mu} v_{c,j_1+k_1-1,\dots,j_n+k_n-1,b}
```

Assumes that:

* `w` is of size `(C,J₁,...,Jₙ,M)`
* `h` is of size `(M,K₁,...,Kₙ,B)`

Here `C` is the channel dimension, `M` is the number of hidden units and `B` is the batch size.
These three dimensions are flat, so that the hidden and batch dimensions must be
flattened before calling `conv_h2v`.
The output `I` is of size `(C, K₁ + J₁ - 1, ..., Kₙ + Jₙ - 1, B)`.

!!! warning
    This is an internal function and is not part of the public API.

!!! warning
    Only works for `n = 1, 2, 3, 4`.
    Due to a technical limitation of Tullio.jl
    ([#129](https://github.com/mcabbott/Tullio.jl/issues/129)),
    `conv_h2v` is defined by hand for different values of `n`.
    Therefore only a small number can be supported.
    If you need some higher value of `n`, consider doing a PR to define the corresponding
    `conv_h2v` method.
"""
function conv_h2v(w::AbstractTensor{N}, h::AbstractTensor{N}) where {N}
    Isz = (size(w,1), (size(w) .+ size(h) .- 1)[2:(end - 1)]..., size(h)[end])
    I = zeros(promote_type(eltype(w), eltype(h)), Isz)
    return conv_h2v!(I, w, h)
end

function conv_h2v!(I::AbstractTensor{N}, w::AbstractTensor{N}, h::AbstractTensor{N}) where {N}
    kernel = size(w)[(begin + 1):(end - 1)]
    output = size(h)[(begin + 1):(end - 1)]
    inputs = kernel .+ output .- 1
    @assert size(I) == (size(w, 1), inputs..., size(h, N))
    @assert size(w) == (size(w, 1), kernel..., size(h, 1))
    @assert size(h) == (size(h, 1), output..., size(h, N))
    for b in axes(h, N), k in CartesianIndices(output), j in CartesianIndices(kernel), μ in axes(h, 1), c in axes(w, 1)
        i = j + k - oneunit(j)
        I[c,i,b] += w[c,j,μ] * h[μ,k,b]
    end
    return I
end

function ChainRulesCore.rrule(::typeof(conv_h2v), w::AbstractTensor{N}, h::AbstractTensor{N}) where {N}
    function conv_h2v_pullback(δI)
        @tullio δw[c,j,μ] = δI[c,k + j - 1,b] * h[μ,k,b]
        return (NoTangent(), δw, NoTangent()) # note we drop gradients of `h`!
    end
    return conv_h2v(w, h), conv_h2v_pullback
end

function _conv_h2v_check_size(I::AbstractTensor{N}, w::AbstractTensor{N}, h::AbstractTensor{N}) where {N}
    input_size = (size(w) .+ size(h) .- 1)[(begin + 1):(end - 1)]
    @assert size(I) == (size(w,1), input_size..., size(h)[end])
    @assert size(w)[end] == size(h, 1) # number of hidden units
end

function conv_h2v!(I::AbstractTensor{3}, w::AbstractTensor{3}, h::AbstractTensor{3}) where {N}
    _conv_h2v_check_size(I, w, h)
    for k ∈ axes(h, 2), μ ∈ axes(h, 1), j ∈ axes(w, 2), c in axes(w, 1)
        i = j + k - 1
        I[c,i,b] = w[c,j,μ] * h[μ,k,b]
    end
end

function conv_h2v!(I::AbstractTensor{4}, w::AbstractTensor{4}, h::AbstractTensor{4}) where {N}
    _conv_h2v_check_size(I, w, h)
    for b ∈ , k ∈ axes(h, 2), j ∈ axes(w, 2)
        i = j + k - 1
        for μ ∈ axes(h, 1), c in axes(w, 1)
            I[c,i,b] += w[c,j,μ] * h[μ,k,b]
        end
    end
end

function _conv_h2v!(I::AbstractTensor{3}, w::AbstractTensor{3}, h::AbstractTensor{3})
    _conv_h2v_check_size(I, w, h)
    @tullio I[c,j+k-1,b] = w[c,j,μ] * h[μ,k,b]
end

function _conv_h2v!(I::AbstractTensor{4}, w::AbstractTensor{4}, h::AbstractTensor{4})
    _conv_h2v_check_size(I, w, h)
    @tullio I[c,j₁+k₁-1,j₂+k₂-1,b] = w[c,j₁,j₂,μ] * h[μ,k₁,k₂,b]
end

function _conv_h2v!(I::AbstractTensor{5}, w::AbstractTensor{5}, h::AbstractTensor{5})
    _conv_h2v_check_size(I, w, h)
    @tullio I[c,j₁+k₁-1,j₂+k₂-1,j₃+k₃-1,b] = w[c,j₁,j₂,j₃,μ] * h[μ,k₁,k₂,k₃,b]
end

function _conv_h2v!(I::AbstractTensor{6}, w::AbstractTensor{6}, h::AbstractTensor{6})
    _conv_h2v_check_size(I, w, h)
    @tullio I[c,j₁+k₁-1,j₂+k₂-1,j₃+k₃-1,j₄+k₄-1,b] = w[c,j₁,j₂,j₃,j₄,μ] * h[μ,k₁,k₂,k₃,k₄,b]
end
