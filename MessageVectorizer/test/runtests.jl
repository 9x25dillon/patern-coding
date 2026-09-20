using MessageVectorizer
using Test

@testset "MessageVectorizer Tests" begin
    @testset "Type Creation" begin
        # Test MotifToken creation
        motif = MotifToken(
            :test_motif,
            Dict{Symbol, Any}(:param1 => 0.5),
            0.8,
            [:temporal, :spatial]
        )
        
        @test motif.name == :test_motif
        @test motif.properties[:param1] == 0.5
        @test motif.weight == 0.8
        @test :temporal in motif.context
        
        # Test MessageState creation
        state = MessageState(
            0,
            [0.1, 0.2, 0.3],
            1.5,
            Dict{Symbol, Float64}(:test_motif => 0.8),
            Dict{String, Any}("test" => "value")
        )
        
        @test length(state.vector_representation) == 3
        @test state.entropy_score == 1.5
    end
    
    @testset "Vectorizer Initialization" begin
        vectorizer = initialize_vectorizer(32)
        
        @test vectorizer.embedding_dim == 32
        @test haskey(vectorizer.symbolic_variables, :state)
        @test haskey(vectorizer.symbolic_variables, :temporal)
        @test haskey(vectorizer.symbolic_variables, :memory)
        @test haskey(vectorizer.symbolic_variables, :spatial)
    end
    
    @testset "Motif Embedding" begin
        vectorizer = initialize_vectorizer(16)
        motif = MotifToken(
            :embedding_test,
            Dict{Symbol, Any}(:value => 0.7),
            0.9,
            [:cognitive]
        )
        
        embedding = create_motif_embedding(motif, 16)
        @test length(embedding) == 16
        @test norm(embedding) <= 1.0  # Should be normalized
        
        add_motif_embedding!(vectorizer, motif)
        @test haskey(vectorizer.motif_embeddings, :embedding_test)
    end
    
    @testset "Symbolic Compression" begin
        vectorizer = initialize_vectorizer(8)
        motif1 = MotifToken(:motif1, Dict{Symbol, Any}(), 0.5, [:temporal])
        motif2 = MotifToken(:motif2, Dict{Symbol, Any}(), 0.3, [:memory])
        
        motifs = [motif1, motif2]
        expr = symbolic_state_compression(motifs, vectorizer)
        
        @test typeof(expr) == typeof(vectorizer.symbolic_variables[:temporal])
    end
    
    @testset "Entropy Computation" begin
        vector = [0.25, 0.25, 0.25, 0.25]
        config = Dict{Symbol, Float64}(:motif1 => 0.5, :motif2 => 0.3)
        
        entropy = compute_entropy(vector, config)
        @test entropy >= 0
        
        # Test with zero vector
        zero_vector = zeros(4)
        zero_entropy = compute_entropy(zero_vector, config)
        @test zero_entropy == 0.0
    end
    
    @testset "Message Vectorization" begin
        vectorizer = initialize_vectorizer(8)
        motif1 = MotifToken(:test1, Dict{Symbol, Any}(:param => 0.6), 0.7, [:temporal])
        motif2 = MotifToken(:test2, Dict{Symbol, Any}(:param => 0.4), 0.5, [:memory])
        
        motifs = [motif1, motif2]
        message_state = vectorize_message(motifs, vectorizer)
        
        @test typeof(message_state) == MessageState
        @test length(message_state.vector_representation) == 8
        @test haskey(message_state.motif_configuration, :test1)
        @test haskey(message_state.motif_configuration, :test2)
        @test haskey(message_state.metadata, "num_motifs")
    end
    
    @testset "AL-ULS Interface" begin
        vectorizer = initialize_vectorizer(4)
        motif = MotifToken(:interface_test, Dict{Symbol, Any}(), 0.6, [:spatial])
        message_state = vectorize_message([motif], vectorizer)
        
        json_output = al_uls_interface(message_state)
        @test typeof(json_output) == String
        @test length(json_output) > 0
    end
end

println("All tests passed!")