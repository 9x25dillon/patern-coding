# Message Vectorizer

## Overview

The Message Vectorizer is a Julia-based system designed to transform motif tokens into higher-order narrative/message states using symbolic computation and vector embeddings. It provides a framework for analyzing and representing complex messages through a combination of symbolic manipulation and numerical techniques.

## Features

- **Symbolic State Representation**: Utilizes Symbolics.jl for symbolic manipulation of motif configurations.
- **Vector Embeddings**: Generates high-dimensional vector representations of motif tokens.
- **Entropy Scoring**: Computes information entropy for message complexity analysis.
- **al-ULS Interface**: Formats output for compatibility with the al-ULS module.
- **Compression**: Efficiently compresses motif configurations into symbolic states.

## Installation

1. Clone this repository.
2. Install Julia dependencies by running the following commands in the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### Basic Usage

To use the Message Vectorizer, you can follow these steps:

1. Create motif tokens using the `MotifToken` structure.
2. Initialize the vectorizer with a specified embedding dimension.
3. Add motif embeddings to the vectorizer.
4. Vectorize a message composed of the motif tokens.
5. Retrieve the output in a format compatible with the al-ULS module.

### Example

Refer to the `examples/message_vectorizer_demo.jl` file for a complete demonstration of the basic usage.

## Running Tests

To ensure the functionality of the Message Vectorizer, run the test suite with the following command:

```bash
julia test/runtests.jl
```

## Docker Support

The Message Vectorizer can be easily deployed using Docker. To build and run the Docker container, use the following command:

```bash
docker-compose up
```

## Contribution

Contributions are welcome! To contribute to the Message Vectorizer project:

1. Fork the repository.
2. Create a feature branch.
3. Add tests for new functionality.
4. Ensure all tests pass.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.