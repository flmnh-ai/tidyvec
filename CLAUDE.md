# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tidyvec is a lightweight vector database and embedding library for R that integrates with the tidyverse ecosystem. It stores vector embeddings alongside data in tibbles, enabling similarity search, multimodal embeddings (text and images), and visualization without leaving the tidyverse workflow.

**Repository:** https://github.com/flmnh-ai/tidyvec
**Documentation:** https://flmnh-ai.github.io/tidyvec/
**Version:** 0.1.0 (experimental)

## Development Commands

### Installation
```r
# Install from GitHub
remotes::install_github("flmnh-ai/tidyvec")

# Python dependencies for neural embeddings are auto-provisioned on first use
```

### Documentation
```r
# Generate documentation from roxygen comments
devtools::document()

# Build pkgdown site locally
pkgdown::build_site()
```

### Testing and Checks
```r
# Run R CMD check (required before submitting PR)
rcmdcheck::rcmdcheck(args = c("--no-manual", "--as-cran"))

# Run tests (uses testthat edition 3)
devtools::test()
```

### Loading for Development
```r
# Load package in development mode
devtools::load_all()
```

## Architecture

### Core Design Pattern

tidyvec uses an **S3 subclass of tibble** with metadata stored as attributes:

```r
tidyvec (extends tibble)
├── Attributes:
│   ├── embedding_column (name of column storing embeddings)
│   └── embedding_fn (function to generate embeddings)
└── Columns:
    └── embedding (list of numeric vectors)
```

### Source Structure (R/ directory)

- **core.R** - Core functionality (`vec()`, `embed()`, `nearest()`, `inspect_collection()`, `write_vec()`, `read_vec()`, `cluster_embeddings()`)
- **embedders.R** - Embedding generators (`embedder_hf()`, `embedder_tfidf()`)
- **visualization.R** - Plotting functions (`viz_embeddings()`, `viz_images()`)
- **imports.R** - Package imports and global variables
- **zzz.R** - Package initialization (`.onLoad()` with Python dependencies)

### Key Implementation Details

1. **Closure-based embedders**: Embedding functions are closures that capture state (e.g., vocabulary for TF-IDF)
2. **Attribute preservation**: The `[.tidyvec` S3 method ensures metadata survives subsetting
3. **Sequential processing**: `embed()` processes items with progress tracking
4. **Lazy evaluation**: Embeddings are NULL until `embed()` is explicitly called
5. **HuggingFace integration**: `embedder_hf()` auto-detects CLIP/SigLIP models and supports multiple devices (cpu, cuda, mps)
6. **Auto-provisioned Python**: Python dependencies are declared in `.onLoad()` and auto-provisioned via reticulate's ephemeral venv
7. **Automatic batching**: HuggingFace embedders support batching (10-50x speedup) via `supports_batch` attribute
8. **Hybrid search**: `nearest()` combines semantic + keyword matching with configurable weights
9. **Persistence**: `write_vec()`/`read_vec()` use qs package for fast save/load

### Dependencies

**Required:** dplyr, tibble, purrr, rlang, ggplot2
**Optional:** reticulate (Python integration), text2vec (TF-IDF), qs (persistence), Rtsne, umap, progress, magick, ggimage

**Python (via reticulate):** torch, transformers, pillow, sentencepiece, protobuf, numpy

## CI/CD

### GitHub Actions Workflows

- **R-CMD-check.yaml**: Runs on Windows, macOS, and Ubuntu (release + devel) on every push/PR
- **pkgdown.yaml**: Builds and deploys documentation site to GitHub Pages

Both workflows must pass before merging PRs.

## Documentation

### Roxygen Comments

All exported functions must have roxygen2 documentation with:
- `@title` and `@description`
- `@param` for each parameter
- `@return` describing return value
- `@examples` with working code examples
- `@export` for exported functions

### Vignettes

Main vignette: `vignettes/getting-started.Rmd`

Covers installation, creating collections, embedding generation (TF-IDF and neural), similarity search, tidyverse integration, image embeddings, and RAG use cases.

## Testing

Currently uses testthat edition 3 (configured in DESCRIPTION). Tests should be placed in `tests/testthat/` directory with filenames matching `test-*.R`.

## Code Style

Follow tidyverse style guide:
- Use `<-` for assignment (not `=`)
- Snake_case for function and variable names
- Spaces around operators
- No trailing whitespace
- Use roxygen2 for all documentation

## Branch Strategy

- **main**: Production branch (used for pkgdown deployment)
- Create feature branches for new work
- PRs must pass R-CMD-check before merging

## Example Images

The package includes example images in `inst/images/` for demonstrating multimodal embeddings:
- Animal images: cat.jpeg, dog.jpeg, dog-on-beach.jpeg
- Landscape images: beach.jpeg, mountain.jpeg, mountain-sunset.jpeg
- Urban: city.jpeg

These can be accessed via `system.file("extdata/images", "filename.jpg", package = "tidyvec")` after installation.
