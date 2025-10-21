#' Calculate Average Precision at k
#'
#' Computes average precision for a single query, measuring how well relevant
#' items are ranked in the retrieval results.
#'
#' @param relevant_ids Character vector of IDs for relevant items (ground truth)
#' @param retrieved_ids Character vector of IDs for retrieved items (ranked by similarity)
#' @param k Number of top results to consider (default: 25)
#'
#' @return Numeric value between 0 and 1, where 1 is perfect retrieval
#'
#' @details
#' Average Precision (AP) measures the precision at each position where a relevant
#' item appears in the ranked results, then averages these precision values.
#' It rewards placing relevant items higher in the ranking.
#'
#' @examples
#' \dontrun{
#' relevant <- c("doc1", "doc2", "doc3")
#' retrieved <- c("doc1", "doc5", "doc2", "doc6", "doc3")
#' ap_at_k(relevant, retrieved, k = 5)  # High score: all relevant items found
#' }
#'
#' @export
ap_at_k <- function(relevant_ids, retrieved_ids, k = 25) {
  if (!is.character(relevant_ids) || !is.character(retrieved_ids)) {
    stop("relevant_ids and retrieved_ids must be character vectors")
  }

  if (k < 1) {
    stop("k must be at least 1")
  }

  # Take top k results
  top_k <- head(retrieved_ids, k)

  # Track precision at each relevant item
  precisions <- numeric(0)
  n_relevant_found <- 0

  for (i in seq_along(top_k)) {
    if (top_k[i] %in% relevant_ids) {
      n_relevant_found <- n_relevant_found + 1
      precision_at_i <- n_relevant_found / i
      precisions <- c(precisions, precision_at_i)
    }
  }

  # Average precision is mean of precisions at relevant positions
  if (length(precisions) == 0) {
    return(0)
  } else {
    return(mean(precisions))
  }
}

#' Calculate Precision at k
#'
#' Computes the proportion of retrieved items that are relevant.
#'
#' @param relevant_ids Character vector of IDs for relevant items
#' @param retrieved_ids Character vector of IDs for retrieved items
#' @param k Number of top results to consider (default: 25)
#'
#' @return Numeric value between 0 and 1
#'
#' @details
#' Precision@k = (# relevant items in top k) / k
#'
#' @examples
#' \dontrun{
#' relevant <- c("doc1", "doc2", "doc3")
#' retrieved <- c("doc1", "doc5", "doc2", "doc6", "doc3")
#' precision_at_k(relevant, retrieved, k = 5)  # 3/5 = 0.6
#' }
#'
#' @export
precision_at_k <- function(relevant_ids, retrieved_ids, k = 25) {
  if (!is.character(relevant_ids) || !is.character(retrieved_ids)) {
    stop("relevant_ids and retrieved_ids must be character vectors")
  }

  if (k < 1) {
    stop("k must be at least 1")
  }

  top_k <- head(retrieved_ids, k)
  n_relevant <- sum(top_k %in% relevant_ids)

  n_relevant / k
}

#' Calculate Recall at k
#'
#' Computes the proportion of relevant items that were retrieved.
#'
#' @param relevant_ids Character vector of IDs for relevant items
#' @param retrieved_ids Character vector of IDs for retrieved items
#' @param k Number of top results to consider (default: 25)
#'
#' @return Numeric value between 0 and 1
#'
#' @details
#' Recall@k = (# relevant items in top k) / (total # relevant items)
#'
#' @examples
#' \dontrun{
#' relevant <- c("doc1", "doc2", "doc3")
#' retrieved <- c("doc1", "doc5", "doc2", "doc6", "doc3")
#' recall_at_k(relevant, retrieved, k = 5)  # 3/3 = 1.0 (all relevant items found)
#' }
#'
#' @export
recall_at_k <- function(relevant_ids, retrieved_ids, k = 25) {
  if (!is.character(relevant_ids) || !is.character(retrieved_ids)) {
    stop("relevant_ids and retrieved_ids must be character vectors")
  }

  if (k < 1) {
    stop("k must be at least 1")
  }

  if (length(relevant_ids) == 0) {
    return(0)
  }

  top_k <- head(retrieved_ids, k)
  n_relevant_found <- sum(top_k %in% relevant_ids)

  n_relevant_found / length(relevant_ids)
}

#' Calculate Mean Average Precision for a tidyvec collection
#'
#' Evaluates retrieval performance by using each item as a query and measuring
#' how well the model retrieves other items with the same attribute value.
#'
#' @param x A tidyvec collection
#' @param variable Column name to evaluate (e.g., "category", "label")
#' @param k Number of results to consider (default: 25)
#' @param id_column Column name for unique identifiers (default: "id")
#' @param similarity_column Column name for similarity scores (default: "similarity")
#' @param exclude_self Whether to exclude the query item from results (default: TRUE)
#' @param progress Whether to show progress bar (default: TRUE)
#'
#' @return Numeric value between 0 and 1 representing Mean Average Precision
#'
#' @details
#' For each item in the collection:
#' 1. Uses its embedding as a query
#' 2. Retrieves k nearest neighbors
#' 3. Calculates AP@k for items with matching variable value
#' 4. Returns the mean of all AP scores
#'
#' Items with missing or "null" values in the specified variable are excluded.
#'
#' @examples
#' \dontrun{
#' library(tidyvec)
#' library(dplyr)
#'
#' # Create a collection with categories
#' data <- tibble(
#'   id = 1:100,
#'   category = sample(c("A", "B", "C"), 100, replace = TRUE),
#'   text = paste("Document", 1:100)
#' )
#'
#' collection <- data %>%
#'   vec(embedding_fn = embedder_tfidf(data$text)) %>%
#'   embed(content_column = "text")
#'
#' # Evaluate how well the model retrieves items by category
#' map_score <- calculate_map(collection, "category", k = 10)
#' print(paste("MAP:", round(map_score, 3)))
#' }
#'
#' @export
calculate_map <- function(x,
                          variable,
                          k = 25,
                          id_column = "id",
                          similarity_column = "similarity",
                          exclude_self = TRUE,
                          progress = TRUE) {
  if (!inherits(x, "tidyvec")) {
    stop("x must be a tidyvec object")
  }

  if (!variable %in% names(x)) {
    stop("Variable '", variable, "' not found in collection")
  }

  if (!id_column %in% names(x)) {
    stop("ID column '", id_column, "' not found in collection")
  }

  emb_col <- embedding_column(x)

  # Setup progress bar
  pb <- NULL
  if (progress && requireNamespace("progress", quietly = TRUE)) {
    pb <- progress::progress_bar$new(
      format = "  Evaluating [:bar] :percent eta: :eta",
      total = nrow(x),
      clear = FALSE
    )
  }

  # For each item, find similar items and calculate AP
  aps <- vapply(seq_len(nrow(x)), function(i) {
    if (!is.null(pb)) pb$tick()

    # Query with this item's embedding
    query_emb <- x[[emb_col]][[i]]
    query_value <- x[[variable]][i]
    query_id <- x[[id_column]][i]

    # Skip if query value is missing
    if (is.na(query_value) || tolower(as.character(query_value)) == "null") {
      return(NA_real_)
    }

    # Find nearest neighbors
    n_retrieve <- if (exclude_self) k + 1 else k
    results <- tryCatch(
      nearest(
        x,
        query_emb,
        as_embedding = TRUE,
        n = n_retrieve
      ),
      error = function(e) {
        warning("Error in nearest() for item ", i, ": ", e$message)
        return(NULL)
      }
    )

    if (is.null(results) || nrow(results) == 0) {
      return(NA_real_)
    }

    # Remove self from results if requested
    if (exclude_self) {
      results <- results[results[[id_column]] != query_id, ]
    }

    # Get relevant IDs (same variable value, excluding self)
    relevant_ids <- x[[id_column]][
      x[[variable]] == query_value & x[[id_column]] != query_id
    ]

    # Calculate AP@k
    ap_at_k(relevant_ids, results[[id_column]], k = k)
  }, numeric(1))

  # Return mean, excluding NAs
  mean(aps, na.rm = TRUE)
}

#' Calculate baseline MAP from category frequencies
#'
#' Computes the expected Mean Average Precision for random retrieval based on
#' the frequency distribution of categories.
#'
#' @param x A tidyvec collection or data frame
#' @param variable Column name to evaluate
#'
#' @return Numeric value representing expected MAP for random retrieval
#'
#' @details
#' The baseline MAP is calculated as the sum of squared category probabilities:
#' baseline = sum(p_i^2) where p_i is the proportion of items in category i.
#'
#' This represents the expected performance if items were retrieved in random order.
#' A model's MAP should exceed this baseline to demonstrate meaningful retrieval.
#'
#' @examples
#' \dontrun{
#' library(tidyvec)
#' library(dplyr)
#'
#' data <- tibble(
#'   id = 1:100,
#'   category = sample(c("A", "B", "C"), 100, replace = TRUE)
#' )
#'
#' baseline <- calculate_baseline(data, "category")
#' print(paste("Random chance MAP:", round(baseline, 3)))
#' }
#'
#' @export
calculate_baseline <- function(x, variable) {
  if (!variable %in% names(x)) {
    stop("Variable '", variable, "' not found in data")
  }

  # Calculate frequency of each category
  freqs <- x %>%
    dplyr::filter(
      !is.na(!!rlang::sym(variable)),
      tolower(as.character(!!rlang::sym(variable))) != "null"
    ) %>%
    dplyr::count(!!rlang::sym(variable)) %>%
    dplyr::mutate(p = .data$n / sum(.data$n))

  # Baseline MAP = sum of (p_i^2)
  sum(freqs$p^2)
}

#' Evaluate retrieval performance across multiple variables
#'
#' Comprehensive evaluation of a tidyvec collection across multiple attributes,
#' computing MAP and comparing to baseline performance.
#'
#' @param x A tidyvec collection
#' @param variables Character vector of column names to evaluate
#' @param k Number of results to consider (default: 25)
#' @param id_column Column name for unique identifiers (default: "id")
#' @param progress Whether to show progress (default: TRUE)
#'
#' @return A tibble with columns: variable, map, baseline, improvement
#'
#' @details
#' For each specified variable:
#' 1. Calculates MAP using calculate_map()
#' 2. Calculates baseline using calculate_baseline()
#' 3. Computes improvement (MAP - baseline)
#'
#' Results are sorted by MAP score (descending).
#'
#' @examples
#' \dontrun{
#' library(tidyvec)
#'
#' # Load a collection
#' collection <- qread("my_collection.qs")
#'
#' # Evaluate multiple attributes
#' results <- evaluate_retrieval(
#'   collection,
#'   variables = c("category", "type", "origin"),
#'   k = 25
#' )
#'
#' print(results)
#' }
#'
#' @export
evaluate_retrieval <- function(x,
                               variables,
                               k = 25,
                               id_column = "id",
                               progress = TRUE) {
  if (!inherits(x, "tidyvec")) {
    stop("x must be a tidyvec object")
  }

  if (!all(variables %in% names(x))) {
    missing <- variables[!variables %in% names(x)]
    stop("Variables not found in collection: ", paste(missing, collapse = ", "))
  }

  if (progress) {
    cat("Evaluating", length(variables), "variables...\n")
  }

  results <- purrr::map_dfr(variables, function(var) {
    if (progress) {
      cat("\n", var, ":\n", sep = "")
    }

    map_score <- calculate_map(
      x,
      variable = var,
      k = k,
      id_column = id_column,
      progress = progress
    )

    baseline <- calculate_baseline(x, var)

    tibble::tibble(
      variable = var,
      map = map_score,
      baseline = baseline,
      improvement = map_score - baseline
    )
  })

  results %>%
    dplyr::arrange(dplyr::desc(.data$map))
}

#' Calculate retrieval metrics for a single query
#'
#' Computes multiple evaluation metrics (AP, Precision, Recall) for a single
#' retrieval result.
#'
#' @param relevant_ids Character vector of IDs for relevant items
#' @param retrieved_ids Character vector of IDs for retrieved items (ranked)
#' @param k Number of top results to consider (default: 25)
#'
#' @return A named list with ap, precision, and recall
#'
#' @examples
#' \dontrun{
#' relevant <- c("doc1", "doc2", "doc3", "doc4")
#' retrieved <- c("doc1", "doc5", "doc2", "doc6", "doc3", "doc7")
#'
#' metrics <- query_metrics(relevant, retrieved, k = 5)
#' print(metrics)
#' # $ap: 0.7667
#' # $precision: 0.6
#' # $recall: 0.75
#' }
#'
#' @export
query_metrics <- function(relevant_ids, retrieved_ids, k = 25) {
  list(
    ap = ap_at_k(relevant_ids, retrieved_ids, k),
    precision = precision_at_k(relevant_ids, retrieved_ids, k),
    recall = recall_at_k(relevant_ids, retrieved_ids, k)
  )
}
