#' Query Similarity Search Across Text and Image Embeddings
#'
#' Performs a multimodal similarity search by comparing a text query against both image and text embeddings.
#' Returns the input data frame sorted by combined similarity score. This is useful for querying datasets
#' where items are described by both visual (image) and textual metadata embeddings.
#'
#' @param query A character string representing the query text (e.g., "porcelain").
#' @param data A data frame or tibble containing the embedding columns. Must include:
#'   - `embedding`: image embeddings (e.g., from a CLIP or SigLIP model)
#'   - `meta_emb`: text (metadata) embeddings.
#' @param model A function that takes a character input (the query) and returns a vector embedding.
#'   This model should be consistent with how the original embeddings in `data` were created.
#' @param text_weight A numeric value between 0 and 1 that controls the relative weight of the text
#'   embedding in the final similarity score. Default is 0.5 (equal weighting of image and text).
#' @param method Similarity metric to use. Options are:
#'   - `"cosine"` (default): cosine similarity
#'   - `"euclidean"`: inverse of Euclidean distance
#'   - `"dot"`: dot product similarity
#'
#' @return A data frame arranged in descending order of combined similarity, with three new columns:
#'   - `sim_img`: similarity between the query and image embedding
#'   - `sim_txt`: similarity between the query and text embedding
#'   - `sim_combined`: weighted average of both similarities


qs <- function(query, data, model, text_weight = 0.5, method = "cosine") {
  sim_fun <- switch(method,
                    cosine = function(a, b) sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2))),
                    euclidean = function(a, b) 1 / (1 + sqrt(sum((a - b)^2))),
                    dot = function(a, b) sum(a * b),
                    stop("Unsupported similarity method")
  )
  
  query_emb <- model(query)
  
  data %>%
    mutate(
      sim_img = purrr::map_dbl(embedding, ~ sim_fun(.x, query_emb)),
      sim_txt = purrr::map_dbl(meta_emb, ~ sim_fun(.x, query_emb)),
      sim_combined = text_weight * sim_txt + (1 - text_weight) * sim_img
    ) %>%
    arrange(desc(sim_combined))
}