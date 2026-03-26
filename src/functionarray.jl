default_eltype() = Float64
default_arraytype() = Array
struct FunctionArray{T,N} <: AbstractArray{T,N}
    f::Function
    domains::Vector{<:AbstractRange}

    FunctionArray(f, ids::Vector) = new{default_eltype(), length(ids)}(f, ids)
end

###### Default case
FunctionArray() = FunctionArray(()->false, Vector{AbstractRange}())
function Base.getindex(FT::FunctionArray{0}, ids...)
    return FT.f()
end

### Single mode constructor
FunctionArray(f, ids::AbstractRange) = FunctionArray(f, [ids,])
FunctionArray(f, ids::AbstractRange...) = FunctionArray(f, [ids...])

### Information about tensor
Base.ndims(FT::FunctionArray) = length(FT.domains)
Base.size(FT::FunctionArray) = Tuple(length.(FT.domains))
dims(FT::FunctionArray) = map(i-> length(i), FT.domains)
dim(FT::FunctionArray, i::Int) = length(FT.domains[i])
    

function convert_ids_to_domain(FT::FunctionArray, ids) 
    return Tuple([domain[i] for (domain, i) in zip(FT.domains, ids)])
end

function Base.getindex(FT::FunctionArray, ids...)
    dids = convert_ids_to_domain(FT, Tuple(ids))
    return FT.f(dids...)
end

function Base.getindex(FT::FunctionArray, ids::CartesianIndex)
    dids = convert_ids_to_domain(FT, Tuple(ids))
    return FT.f(dids...)
end

function materialize(FT::FunctionArray)
    tensor = default_arraytype(){eltype(FT)}(undef, size(FT))
    map!(x->getindex(FT, x), tensor, CartesianIndices(tensor))

    return tensor
end