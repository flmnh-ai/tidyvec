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
  modality <- match.arg(modality)

  # Check if this is a SigLIP model
  is_siglip <- grepl("siglip", tolower(model_name))
  is_siglip2 <- grepl("siglip2", tolower(model_name))

  # Helper to load SigLIP models
  load_siglip_model <- function(model_name, cache_dir, device, transformers) {
    model <- transformers$SiglipModel$from_pretrained(model_name, cache_dir = cache_dir)
    processor <- transformers$AutoProcessor$from_pretrained(model_name, cache_dir = cache_dir)
    model$to(device)
    model$eval()
    list(model = model, processor = processor)
  }

  # Import Python modules (dependencies auto-provisioned via py_require() in .onLoad())
  transformers <- reticulate::import("transformers")
  torch <- reticulate::import("torch")
  PIL <- reticulate::import("PIL")

  # Setup model based on modality
  if (modality == "multimodal") {
    if (is_siglip) {
      # For SigLIP models
      loaded <- load_siglip_model(model_name, cache_dir, device, transformers)
      model <- loaded$model
      processor <- loaded$processor

      function(x) {
        # Detect if x is an image path or text
        if (is.character(x) && length(x) == 1 && file.exists(x) && !dir.exists(x)) {
          # Process as image
          image <- PIL$Image$open(x)
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
        if (is.character(x) && length(x) == 1 && file.exists(x) && !dir.exists(x)) {
          # Process as image
          image <- PIL$Image$open(x)
          inputs <- processor(images = list(image), return_tensors = "pt")
          inputs$to(device)

          with(torch$no_grad(), {
            outputs <- model$get_image_features(inputs$pixel_values)
            # Normalize embeddings
            outputs <- outputs / outputs$norm(dim = -1L, keepdim = TRUE)
          })
        } else {
          # Process as text
          inputs <- processor(text = list(as.character(x)), return_tensors = "pt")
          inputs$to(device)

          with(torch$no_grad(), {
            outputs <- model$get_text_features(inputs$input_ids)
            # Normalize embeddings
            outputs <- outputs / outputs$norm(dim = -1L, keepdim = TRUE)
          })
        }

        # Convert to R vector
        as.numeric(outputs$cpu()$numpy()[1, ])
      }
    }
  } else if (modality == "text") {
    # Text-only model
    if (is_siglip) {
      # For SigLIP models with text-only usage
      loaded <- load_siglip_model(model_name, cache_dir, device, transformers)
      model <- loaded$model
      processor <- loaded$processor

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
          outputs <- model(inputs$input_ids, inputs$attention_mask)

          # Extract embedding
          if ("pooler_output" %in% names(outputs)) {
            emb_tensor <- outputs$pooler_output
          } else {
            # Use mean pooling as fallback
            emb_tensor <- torch$mean(outputs$last_hidden_state[1], dim = 0L, keepdim = TRUE)
          }

          # Normalize
          emb_tensor <- emb_tensor / emb_tensor$norm(dim = -1L, keepdim = TRUE)
        })

        # Convert to R vector
        as.numeric(emb_tensor$cpu()$numpy()[1, ])
      }
    }
  } else {  # image
    if (is_siglip) {
      # SigLIP image-only model
      loaded <- load_siglip_model(model_name, cache_dir, device, transformers)
      model <- loaded$model
      processor <- loaded$processor

      function(x) {
        # Assume x is a path to an image
        image <- PIL$Image$open(x)
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
        image <- PIL$Image$open(x)
        inputs <- processor(images = list(image), return_tensors = "pt")
        inputs$to(device)

        with(torch$no_grad(), {
          outputs <- model(inputs$pixel_values)

          # Extract embedding
          if ("pooler_output" %in% names(outputs)) {
            emb_tensor <- outputs$pooler_output
          } else {
            # Use global average pooling as fallback
            emb_tensor <- torch$mean(outputs$last_hidden_state, dims = c(1L, 2L), keepdim = TRUE)
          }

          # Normalize
          emb_tensor <- emb_tensor / emb_tensor$norm(dim = -1L, keepdim = TRUE)
        })

        # Convert to R vector
        as.numeric(emb_tensor$cpu()$numpy()[1, ])
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
      stop("Query contains no terms found in corpus vocabulary")
    }

    tfidf_vec <- tfidf$transform(dtm_new)
    as.numeric(tfidf_vec[1, ])
  }
}