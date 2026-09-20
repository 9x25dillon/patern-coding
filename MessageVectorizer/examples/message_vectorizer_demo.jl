# Message Vectorizer Demo

using MessageVectorizer

# Create motif tokens
isolation_motif = MotifToken(
    :isolation_time,
    Dict{Symbol, Any}(:intensity => 0.8, :duration => 24.0),
    0.7,
    [:temporal, :spatial, :emotional]
)

decay_motif = MotifToken(
    :decay_memory,
    Dict{Symbol, Any}(:decay_rate => 0.3, :memory_strength => 0.6),
    0.6,
    [:cognitive, :temporal, :neural]
)

println("Created motif tokens:")
println("  Isolation motif: ", isolation_motif.name)
println("  Decay motif: ", decay_motif.name)

# Initialize vectorizer
vectorizer = initialize_vectorizer(64)

println("\nInitialized vectorizer with embedding dimension: ", vectorizer.embedding_dim)

# Add motif embeddings
add_motif_embedding!(vectorizer, isolation_motif)
add_motif_embedding!(vectorizer, decay_motif)

println("\nAdded motif embeddings:")
println("  Isolation embedding length: ", length(vectorizer.motif_embeddings[:isolation_time]))
println("  Decay embedding length: ", length(vectorizer.motif_embeddings[:decay_memory]))

# Vectorize message
motifs = [isolation_motif, decay_motif]
message_state = vectorize_message(motifs, vectorizer)

println("\nVectorized message:")
println("  Symbolic expression: ", message_state.symbolic_expression)
println("  Vector length: ", length(message_state.vector_representation))
println("  Entropy score: ", message_state.entropy_score)
println("  Number of motifs: ", message_state.metadata["num_motifs"])

# Get al-ULS compatible output
uls_output = al_uls_interface(message_state)
println("\nAL-ULS interface output:")
println(uls_output)

# Demonstrate advanced configuration
println("\n--- Advanced Configuration ---")
advanced_vectorizer = initialize_vectorizer(
    128,                    # embedding dimension
    entropy_threshold=0.7,  # entropy threshold
    compression_ratio=0.85  # compression ratio
)

println("Advanced vectorizer created:")
println("  Embedding dimension: ", advanced_vectorizer.embedding_dim)
println("  Entropy threshold: ", advanced_vectorizer.entropy_threshold)
println("  Compression ratio: ", advanced_vectorizer.compression_ratio)