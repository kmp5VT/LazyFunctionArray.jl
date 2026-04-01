module LazyFunctionArray
    using Base.Threads

    include("functionarray.jl")
    export FunctionArray, dim, make_tensor
end