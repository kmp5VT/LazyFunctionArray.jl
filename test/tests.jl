using Pkg, Test, LinearAlgebra
# Pkg.develop(path="$(@__DIR__)/LazyFunctionTensor.jl/")
# Pkg.activate()
using LazyFunctionArray
using LazyFunctionArray: FunctionArray
ft_empty = FunctionArray()
@test ft_empty[] == false

function g(x,y)
    return x.^2 - sin(y)
end

FT = FunctionArray(g, [-1:.1:1, -2:2])
@test LazyFunctionArray.dims(FT) == [21,5]
@test FT[3,3] ≈ 0.64

@test LazyFunctionArray.convert_ids_to_domain(FT, [1,3]) == (-1, 0)

t = Array{Float64}(undef, 21,5)
for i in 1:21
    for j in 1:5
        x = (-1:0.1:1)[i]
        y = (-2:2)[j]
        t[i,j] = g(x,y)
    end
end

@test t ≈ LazyFunctionArray.materialize(FT)

m = randn(21,5);
@test m + FT ≈ m + t
@test FT * m' ≈ t * m'

q,r = qr(FT)
@test q * r ≈ t