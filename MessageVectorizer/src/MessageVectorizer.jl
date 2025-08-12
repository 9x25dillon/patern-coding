module MessageVectorizer

using Symbolics
using LinearAlgebra
using StatsBase
using JSON3
using DataFrames
using Random

export MotifToken, MessageState, MessageVectorizer
export vectorize_message, compute_entropy, create_motif_embedding
export symbolic_state_compression, al_uls_interface, add_motif_embedding!
export initialize_vectorizer

"""
    MotifToken

Represents a basic motif token with symbolic properties.

# Fields
- `name::Symbol`: Motif identifier
- `properties::Dict{Symbol, Any}`: Motif properties
- `weight::Float64`: Motif weight
- `context::Vector{Symbol}`: Contextual tags
"""
struct MotifToken
    name::Symbol
    properties::Dict{Symbol, Any}
    weight::Float64
    context::Vector{Symbol}
end

"""
    MessageState

Represents a compressed symbolic state of a message.

# Fields
- `symbolic_expression::Num`: Symbolic representation
- `vector_representation::Vector{Float64}`: Vector embedding
- `entropy_score::Float64`: Information entropy
- `motif_configuration::Dict{Symbol, Float64}`: Motif weights
- `metadata::Dict{String, Any}`: Additional metadata
"""
struct MessageState
    symbolic_expression::Num
    vector_representation::Vector{Float64}
    entropy_score::Float64
    motif_configuration::Dict{Symbol, Float64}
    metadata::Dict{String, Any}
end

"""
    MessageVectorizer

Main vectorizer for transforming motif tokens.

# Fields
- `motif_embeddings::Dict{Symbol, Vector{Float64}}`: Stored embeddings
- `symbolic_variables::Dict{Symbol, Num}`: Symbolic variables
- `embedding_dim::Int`: Embedding dimension
- `entropy_threshold::Float64`: Entropy threshold
- `compression_ratio::Float64`: Compression ratio
"""
mutable struct MessageVectorizer
    motif_embeddings::Dict{Symbol, Vector{Float64}}
    symbolic_variables::Dict{Symbol, Num}
    embedding_dim::Int
    entropy_threshold::Float64
    compression_ratio::Float64
end

"""
    initialize_vectorizer(dim::Int; entropy_threshold=0.5, compression_ratio=0.8)

Create and initialize a MessageVectorizer with symbolic variables.

# Arguments
- `dim::Int`: Embedding dimension
- `entropy_threshold::Float64`: Entropy threshold (default: 0.5)
- `compression_ratio::Float64`: Compression ratio (default: 0.8)

# Returns
- `MessageVectorizer`: Initialized vectorizer
"""
function initialize_vectorizer(dim::Int; entropy_threshold=0.5, compression_ratio=0.8)
    @variables s τ μ σ
    
    symbolic_vars = Dict{Symbol, Num}(
        :state => s,
        :temporal => τ,
        :memory => μ,
        :spatial => σ
    )
    
    return MessageVectorizer(
        Dict{Symbol, Vector{Float64}}(),
        symbolic_vars,
        dim,
        entropy_threshold,
        compression_ratio
    )
end

"""
    create_motif_embedding(motif::MotifToken, dim::Int)

Create a vector embedding for a motif token.

# Arguments
- `motif::MotifToken`: Motif token to embed
- `dim::Int`: Embedding dimension

# Returns
- `Vector{Float64}`: Vector embedding
"""
function create_motif_embedding(motif::MotifToken, dim::Int)
    Random.seed!(hash(motif.name))
    
    embedding = zeros(Float64, dim)
    base_value = motif.weight
    
    for (key, value) in motif.properties
        if isa(value, Number)
            prop_influence = Float64(value) * base_value
            for i in 1:min(dim, 10)
                embedding[i] += prop_influence * rand()
            end
        end
    end
    
    if norm(embedding) > 0
        embedding = embedding / norm(embedding)
    end
    
    return embedding
end

"""
    add_motif_embedding!(vectorizer::MessageVectorizer, motif::MotifToken)

Add a motif embedding to the vectorizer.

# Arguments
- `vectorizer::MessageVectorizer`: Vectorizer to update
- `motif::MotifToken`: Motif token to embed
"""
function add_motif_embedding!(vectorizer::MessageVectorizer, motif::MotifToken)
    embedding = create_motif_embedding(motif, vectorizer.embedding_dim)
    vectorizer.motif_embeddings[motif.name] = embedding
end

"""
    symbolic_state_compression(motifs::Vector{MotifToken}, vectorizer::MessageVectorizer)

Compress motif tokens into a symbolic state representation.

# Arguments
- `motifs::Vector{MotifToken}`: Motif tokens to compress
- `vectorizer::MessageVectorizer`: Vectorizer with symbolic variables

# Returns
- `Num`: Symbolic expression
"""
function symbolic_state_compression(motifs::Vector{MotifToken}, vectorizer::MessageVectorizer)
    expr = 0
    vars = vectorizer.symbolic_variables
    
    for motif in motifs
        coefficient = motif.weight
        
        if :temporal in motif.context
            expr += coefficient * vars[:temporal]
        elseif :memory in motif.context
            expr += coefficient * vars[:memory]
        elseif :spatial in motif.context
            expr += coefficient * vars[:spatial]
        else
            expr += coefficient * vars[:state]
        end
    end
    
    return expr
end

"""
    compute_entropy(vector::Vector{Float64}, motif_config::Dict{Symbol, Float64})

Compute entropy score for a message vector.

# Arguments
- `vector::Vector{Float64}`: Vector representation
- `motif_config::Dict{Symbol, Float64}`: Motif configuration

# Returns
- `Float64`: Entropy score
"""
function compute_entropy(vector::Vector{Float64}, motif_config::Dict{Symbol, Float64})
    if sum(abs.(vector)) == 0
        return 0.0
    end
    
    normalized = abs.(vector) / sum(abs.(vector))
    non_zero = normalized[normalized .> 1e-10]
    
    if length(non_zero) == 0
        return 0.0
    end
    
    entropy = -sum(non_zero .* log.(non_zero))
    
    num_motifs = length(motif_config)
    if num_motifs > 0
        entropy *= log(num_motifs + 1)
    end
    
    return entropy
end

"""
    vectorize_message(motifs::Vector{MotifToken}, vectorizer::MessageVectorizer)

Transform motif tokens into a message state vector.

# Arguments
- `motifs::Vector{MotifToken}`: Motif tokens to vectorize
- `vectorizer::MessageVectorizer`: Vectorizer to use

# Returns
- `MessageState`: Compressed message state
"""
function vectorize_message(motifs::Vector{MotifToken}, vectorizer::MessageVectorizer)
    motif_config = Dict{Symbol, Float64}()
    for motif in motifs
        motif_config[motif.name] = motif.weight
    end
    
    combined_vector = zeros(Float64, vectorizer.embedding_dim)
    for motif in motifs
        if haskey(vectorizer.motif_embeddings, motif.name)
            embedding = vectorizer.motif_embeddings[motif.name]
            combined_vector += motif.weight * embedding
        else
            embedding = create_motif_embedding(motif, vectorizer.embedding_dim)
            combined_vector += motif.weight * embedding
        end
    end
    
    if norm(combined_vector) > 0
        combined_vector = combined_vector / norm(combined_vector)
    end
    
    symbolic_expr = symbolic_state_compression(motifs, vectorizer)
    entropy_score = compute_entropy(combined_vector, motif_config)
    
    metadata = Dict{String, Any}(
        "num_motifs" => length(motifs),
        "compression_ratio" => vectorizer.compression_ratio,
        "timestamp" => time(),
        "compressed_size" => length(combined_vector),
        "information_density" => entropy_score / max(length(combined_vector), 1)
    )
    
    return MessageState(
        symbolic_expr,
        combined_vector,
        entropy_score,
        motif_config,
        metadata
    )
end

"""
    al_uls_interface(message_state::MessageState)

Format message state for al-ULS module consumption.

# Arguments
- `message_state::MessageState`: Message state to format

# Returns
- `String`: JSON formatted string
"""
function al_uls_interface(message_state::MessageState)
    symbolic_str = string(message_state.symbolic_expression)
    
    output = Dict{String, Any}(
        "symbolic_expression" => symbolic_str,
        "vector_representation" => message_state.vector_representation,
        "entropy_score" => message_state.entropy_score,
        "motif_configuration" => message_state.motif_configuration,
        "metadata" => message_state.metadata,
        "compressed_size" => length(message_state.vector_representation),
        "information_density" => message_state.metadata["information_density"]
    )
    
    return JSON3.write(output)
end

end # module