.onLoad <- function(libname, pkgname) {
  # Disable tokenizers parallelism to avoid forking warnings
  Sys.setenv("TOKENIZERS_PARALLELISM" = "false")

  # Enable PyTorch MPS fallback for macOS compatibility
  # Allows graceful fallback to CPU for unsupported MPS operations
  Sys.setenv("PYTORCH_ENABLE_MPS_FALLBACK" = "1")

  # Declare Python requirements for HuggingFace embedders
  # These will be auto-provisioned in an ephemeral venv when Python initializes
  reticulate::py_require(c(
    "torch",
    "transformers",
    "pillow",
    "sentencepiece",
    "protobuf",
    "numpy"
  ))
}
