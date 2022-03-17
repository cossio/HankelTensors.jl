using Test: @testset, @test, @inferred
using HankelTensors: Hankel, hankel

@testset "hankel" begin
    channel_size = (3,2)
    input_size = (5,5)
    kernel_size = (3,2)
    batch_size = (2,)
    v = randn(channel_size..., input_size..., batch_size...)
    A = @inferred Hankel(v, channel_size, kernel_size)
    @test (@inferred size(A)) == (channel_size..., kernel_size..., (input_size .- kernel_size .+ 1)..., batch_size...)
    @test A == @inferred hankel(v, channel_size, kernel_size)
    @test ndims(A) == ndims(v) + length(kernel_size)
    for c in CartesianIndices(channel_size), j in CartesianIndices(kernel_size), k in CartesianIndices(input_size .- kernel_size .+ 1), b in CartesianIndices(batch_size)
        i = CartesianIndex(Tuple(j) .+ Tuple(k) .- 1)
        @test A[c,j,k,b] == v[c,i,b]
        #@inferred A[c,j,k,b]  # https://github.com/JuliaLang/julia/issues/44059
    end
end
