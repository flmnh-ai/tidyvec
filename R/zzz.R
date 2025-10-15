.onLoad <- function(libname, pkgname) {
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
