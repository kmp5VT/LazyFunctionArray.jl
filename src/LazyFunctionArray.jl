module LazyFunctionArray
    using Base.Threads

    include("functionarray.jl")
    export FunctionTensor, dim, make_tensor
end