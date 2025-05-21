#' Create a HuggingFace embedding function
#'
#' @param model_name Name of the HuggingFace model
#' @param modality Type of model ("text", "image", or "multimodal")
#' @param device Device to use ("cpu", "cuda", or "mps")
#' @param cache_dir Optional directory for caching model files
#' @return An embedding function
#' @export
embedder_hf <- function(model_name,
                        modality = c("multimodal", "text", "image"),
                        device = "cpu",
                        cache_dir = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for HuggingFace integration")
  }

  modality <- match.arg(modality)

  # Check if this is a SigLIP model
  is_siglip <- grepl("siglip", tolower(model_name))
  is_siglip2 <- grepl("siglip2", tolower(model_name))

  # Add debug information about detected model type
  if (is_siglip) {
    message("SigLIP model detected: ", model_name)
    if (is_siglip2) {
      message("SigLIP-2 variant detected, will use max_length=64 for text")
    }
  }

  # Initialize Python environment
  reticulate::py_run_string(paste0("
import torch
import transformers
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
", if(is_siglip) "from transformers import SiglipModel" else "", "
device = '", device, "'
"))

  # Import needed modules
  transformers <- reticulate::import("transformers")
  torch <- reticulate::import("torch")
  PIL <- reticulate::import("PIL")

  # Setup model based on modality
  if (modality == "multimodal") {
    if (is_siglip) {
      # For SigLIP models
      model <- transformers$SiglipModel$from_pretrained(model_name, cache_dir = cache_dir)
      processor <- transformers$AutoProcessor$from_pretrained(model_name, cache_dir = cache_dir)
      model$to(device)
      model$eval()  # Set to evaluation mode

      function(x) {
        # Detect if x is an image path or text
        if (is.character(x) && length(x) == 1 && grepl("\\.(jpg|jpeg|png|gif|bmp)$", x)) {
          # Process as image
          image <- PIL$Image$open(x)$convert("RGB")
          inputs <- processor(images = list(image), return_tensors = "pt")
          inputs$to(device)

          with(torch$no_grad(), {
            image_features <- model$get_image_features(inputs$pixel_values)
            # Normalize embeddings
            image_features <- image_features / image_features$norm(dim = -1L, keepdim = TRUE)
          })

          # Convert to R vector
          emb <- as.numeric(image_features$cpu()$numpy()[1, ])
        } else {
          # Process as text
          text_kwargs <- list(
            text = list(as.character(x)),
            return_tensors = "pt",
            padding = "max_length"
          )

          if (is_siglip2) {
            text_kwargs$max_length <- 64L
          }

          inputs <- do.call(processor, text_kwargs)
          inputs$to(device)

          with(torch$no_grad(), {
            # Get text features - only pass attention_mask if it exists
            if ("attention_mask" %in% names(inputs)) {
              text_features <- model$get_text_features(inputs$input_ids,
                                                       attention_mask = inputs$attention_mask)
            } else {
              text_features <- model$get_text_features(inputs$input_ids)
            }
            # Normalize embeddings
            text_features <- text_features / text_features$norm(dim = -1L, keepdim = TRUE)
          })

          # Convert to R vector
          emb <- as.numeric(text_features$cpu()$numpy()[1, ])
        }

        # Return normalized embedding
        emb
      }
    } else {
      # For CLIP-like models (existing code)
      model <- transformers$CLIPModel$from_pretrained(model_name, cache_dir = cache_dir)
      processor <- transformers$CLIPProcessor$from_pretrained(model_name, cache_dir = cache_dir)
      model$to(device)

      function(x) {
        # Detect if x is an image path or text
        if (is.character(x) && length(x) == 1 && grepl("\\.(jpg|jpeg|png|gif|bmp)$", x)) {
          # Process as image
          image <- PIL$Image$open(x)$convert("RGB")
          inputs <- processor(images = list(image), return_tensors = "pt")
          inputs$to(device)

          with(torch$no_grad(), {
            # Use direct attribute access instead of ** unpacking
            outputs <- model$get_image_features(inputs$pixel_values)
          })
        } else {
          # Process as text
          inputs <- processor(text = list(as.character(x)), return_tensors = "pt")
          inputs$to(device)

          with(torch$no_grad(), {
            # Use direct attribute access instead of ** unpacking
            outputs <- model$get_text_features(inputs$input_ids)
          })
        }

        # Convert to R vector and normalize
        emb <- as.numeric(outputs$cpu()$numpy()[1, ])
        emb / sqrt(sum(emb^2))  # L2 normalization
      }
    }
  } else if (modality == "text") {
    # Text-only model
    if (is_siglip) {
      # For SigLIP models with text-only usage
      model <- transformers$SiglipModel$from_pretrained(model_name, cache_dir = cache_dir)
      processor <- transformers$AutoProcessor$from_pretrained(model_name, cache_dir = cache_dir)
      model$to(device)
      model$eval()

      function(x) {
        text_kwargs <- list(
          text = list(as.character(x)),
          return_tensors = "pt",
          padding = "max_length"
        )

        if (is_siglip2) {
          text_kwargs$max_length <- 64L
        }

        inputs <- do.call(processor, text_kwargs)
        inputs$to(device)

        with(torch$no_grad(), {
          # Get text features - only pass attention_mask if it exists
          if ("attention_mask" %in% names(inputs)) {
            text_features <- model$get_text_features(inputs$input_ids,
                                                     attention_mask = inputs$attention_mask)
          } else {
            text_features <- model$get_text_features(inputs$input_ids)
          }
          # Normalize embeddings
          text_features <- text_features / text_features$norm(dim = -1L, keepdim = TRUE)
        })

        # Convert to R vector
        as.numeric(text_features$cpu()$numpy()[1, ])
      }
    } else {
      # Text-only model (existing code)
      model <- transformers$AutoModel$from_pretrained(model_name, cache_dir = cache_dir)
      tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name, cache_dir = cache_dir)
      model$to(device)

      function(x) {
        inputs <- tokenizer(as.character(x), return_tensors = "pt", truncation = TRUE)
        inputs$to(device)

        with(torch$no_grad(), {
          # Replace py_call_object with direct method call
          outputs <- model(inputs$input_ids, inputs$attention_mask)
        })

        # Rest of the function remains the same
        if ("pooler_output" %in% names(outputs)) {
          emb <- as.numeric(outputs$pooler_output$cpu()$numpy()[1, ])
        } else {
          # Use mean pooling as fallback
          emb <- as.numeric(torch$mean(outputs$last_hidden_state[1], dim = 0L)$cpu()$numpy())
        }

        # Normalize
        emb / sqrt(sum(emb^2))
      }
    }
  } else {  # image
    if (is_siglip) {
      # SigLIP image-only model
      model <- transformers$SiglipModel$from_pretrained(model_name, cache_dir = cache_dir)
      processor <- transformers$AutoProcessor$from_pretrained(model_name, cache_dir = cache_dir)
      model$to(device)
      model$eval()

      function(x) {
        # Assume x is a path to an image
        image <- PIL$Image$open(x)$convert("RGB")
        inputs <- processor(images = list(image), return_tensors = "pt")
        inputs$to(device)

        with(torch$no_grad(), {
          image_features <- model$get_image_features(inputs$pixel_values)
          # Normalize embeddings
          image_features <- image_features / image_features$norm(dim = -1L, keepdim = TRUE)
        })

        # Convert to R vector
        as.numeric(image_features$cpu()$numpy()[1, ])
      }
    } else {
      # Image-only model (existing code)
      model <- transformers$AutoModel$from_pretrained(model_name, cache_dir = cache_dir)
      processor <- transformers$AutoImageProcessor$from_pretrained(model_name, cache_dir = cache_dir)
      model$to(device)

      # And in the image-only model function section:
      function(x) {
        # Assume x is a path to an image
        image <- PIL$Image$open(x)$convert("RGB")
        inputs <- processor(images = list(image), return_tensors = "pt")
        inputs$to(device)

        with(torch$no_grad(), {
          # Replace py_call_object with direct method call
          outputs <- model(inputs$pixel_values)
        })

        # Extract embedding
        if ("pooler_output" %in% names(outputs)) {
          emb <- as.numeric(outputs$pooler_output$cpu()$numpy()[1, ])
        } else {
          # Use global average pooling as fallback
          emb <- as.numeric(torch$mean(outputs$last_hidden_state, dims = c(1L, 2L))$cpu()$numpy()[1, ])
        }

        # Normalize
        emb / sqrt(sum(emb^2))
      }
    }
  }
}

#' Create a simple TF-IDF embedding function for text
#'
#' @param corpus Text corpus to build vocabulary
#' @param min_freq Minimum term frequency
#' @return An embedding function
#' @export
embedder_tfidf <- function(corpus, min_freq = 2) {
  if (!requireNamespace("text2vec", quietly = TRUE)) {
    stop("Package 'text2vec' is required")
  }

  # Process corpus
  it <- text2vec::itoken(corpus, preprocessor = tolower,
                         tokenizer = text2vec::word_tokenizer)
  vocab <- text2vec::create_vocabulary(it)
  pruned_vocab <- text2vec::prune_vocabulary(vocab, term_count_min = min_freq)

  # Create vectorizer and tfidf model
  vectorizer <- text2vec::vocab_vectorizer(pruned_vocab)
  dtm <- text2vec::create_dtm(it, vectorizer)
  tfidf <- text2vec::TfIdf$new()
  tfidf$fit_transform(dtm)

  # Return embedding function
  function(x) {
    it_new <- text2vec::itoken(as.character(x), preprocessor = tolower,
                               tokenizer = text2vec::word_tokenizer)
    dtm_new <- text2vec::create_dtm(it_new, vectorizer)

    # Check if dtm is empty
    if (nrow(dtm_new) == 0 || sum(dtm_new) == 0) {
      warning("Query contains no terms found in corpus vocabulary")
      return(NULL)  # Return NULL instead of an empty vector
    }

    tfidf_vec <- tfidf$transform(dtm_new)
    as.numeric(tfidf_vec[1, ])
  }
}

#' Setup Python environment for TidyVec
#'
#' @param method Method for environment creation, either "virtualenv" or "conda"
#' @param envname Name of the environment to create
#' @param packages Additional packages to install
#' @return Invisible TRUE if successful
#' @export
setup_python <- function(method = c("virtualenv", "conda"),
                         envname = "tidyvec_env",
                         packages = c("torch", "transformers", "pillow", "sentencepiece", "protobuf", "numpy")) {
  method <- match.arg(method)

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required")
  }

  if (method == "virtualenv") {
    if (!reticulate::virtualenv_exists(envname)) {
      message("Creating virtual environment '", envname, "'...")
      reticulate::virtualenv_create(envname)
      message("Installing required packages (this may take a while)...")
      reticulate::py_install(packages, envname = envname)
    }
    reticulate::use_virtualenv(envname)
  } else {
    # Check if conda is available more safely
    conda_available <- tryCatch({
      reticulate::py_run_string("import conda")
      TRUE
    }, error = function(e) FALSE)

    if (!conda_available) {
      message("Installing miniconda...")
      reticulate::install_miniconda()
    }

    # Check if conda environment exists more safely
    conda_env_exists <- function(envname) {
      tryCatch({
        envs <- reticulate::conda_list()
        envname %in% envs$name
      }, error = function(e) FALSE)
    }

    if (!conda_env_exists(envname)) {
      message("Creating conda environment '", envname, "'...")
      reticulate::conda_create(envname)
      message("Installing required packages (this may take a while)...")
      reticulate::conda_install(envname, packages)
    }

    reticulate::use_condaenv(envname)
  }

  message("Python environment setup complete. Using ", reticulate::py_config()$python)
  invisible(TRUE)
}
