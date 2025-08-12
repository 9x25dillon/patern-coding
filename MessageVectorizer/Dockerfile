FROM julia:1.9

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

# Run tests to verify installation
RUN julia test/runtests.jl

# Expose port (if needed for future web interface)
EXPOSE 8080

# Default command
CMD ["julia", "examples/message_vectorizer_demo.jl"]