default_eltype() = Float64
default_arraytype() = Array
struct FunctionArray{T,N} <: AbstractArray{T,N}
    f::Function
    domains::Vector{<:AbstractRange}

    FunctionArray(f, ids::Vector) = new{default_eltype(), length(ids)}(f, ids)
end

###### Default case
FunctionArray() = FunctionArray(()->false, Vector{AbstractRange}())
function Base.getindex(FA::FunctionArray{0}, ids...)
    return FA.f()
end

### Single mode constructor
FunctionArray(f, ids::AbstractRange) = FunctionArray(f, [ids,])
FunctionArray(f, ids::AbstractRange...) = FunctionArray(f, [ids...])

## If trying to build with no information (undef) just make array for now.
FunctionArray(::UndefInitializer, ids::Tuple) = default_arraytype(){default_eltype}(undef, ids)
FunctionArray{N,T}(::UndefInitializer, ids::Tuple) where{N,T} = default_arraytype(){N,T}(undef, ids)

### Information about tensor
Base.ndims(FA::FunctionArray) = length(FA.domains)
Base.size(FA::FunctionArray) = Tuple(length.(FA.domains))
## Similar will materialize the function because we don't know what 
## function the new tensor will have
Base.similar(FA::FunctionArray) = similar(default_arraytype(){eltype(FA)}, Tuple(dims(FA)))
Base.similar(FA::FunctionArray, ds::Tuple) = similar(default_arraytype(){eltype(FA)}, ds)
dims(FA::FunctionArray) = map(i-> length(i), FA.domains)
dim(FA::FunctionArray, i::Int) = length(FA.domains[i])
    
func(FA::FunctionArray) = FA.f
domains(FA::FunctionArray) = FA.domains

function convert_ids_to_domain(FA::FunctionArray, ids) 
    return map((domain, i) -> domain[i], FA.domains, ids)
end

function Base.getindex(FA::FunctionArray, ids...)
    dids = convert_ids_to_domain(FA, Tuple(ids))
    return FA.f(dids...)
end

function Base.getindex(FA::FunctionArray, ids::CartesianIndex)
    dids = convert_ids_to_domain(FA, Tuple(ids))
    return FA.f(dids...)
end

function materialize(FA::FunctionArray)
    tensor = default_arraytype(){eltype(FA)}(undef, size(FA))
    map!(x->getindex(FA, x), tensor, CartesianIndices(tensor))

    return tensor
end