#' Creates a vector collection from a data frame or tibble
#'
#' @param x A data frame or tibble
#' @param embedding_column Name of column containing embeddings (or to be created)
#' @param embedding_fn Function to generate embeddings (optional)
#' @return A tidyvec object
#' @export
vec <- function(x, embedding_column = "embedding", embedding_fn = NULL) {
  # Ensure x is a tibble
  x <- tibble::as_tibble(x)

  # Ensure embedding column exists
  if (!embedding_column %in% names(x)) {
    # Create a list of NULL values with the same length as the number of rows in x
    x[[embedding_column]] <- vector("list", nrow(x))
  }

  # Create tidyvec object
  class(x) <- c("tidyvec", class(x))

  # Store embedding column name and function
  attr(x, "embedding_column") <- embedding_column
  attr(x, "embedding_fn") <- embedding_fn

  x
}

#' Extract embedding column from a tidyvec object
#'
#' @param x A tidyvec object
#' @return Name of the embedding column
#' @keywords internal
embedding_column <- function(x) {
  if (!inherits(x, "tidyvec")) {
    stop("Not a tidyvec object")
  }
  attr(x, "embedding_column", exact = TRUE) %||% "embedding"
}

#' Extract embedding function from a tidyvec object
#'
#' @param x A tidyvec object
#' @return The embedding function or NULL
#' @keywords internal
embedding_fn <- function(x) {
  if (!inherits(x, "tidyvec")) {
    stop("Not a tidyvec object")
  }
  attr(x, "embedding_fn", exact = TRUE)
}

#' @export
print.tidyvec <- function(x, ...) {
  emb_col <- embedding_column(x)
  has_fn <- !is.null(embedding_fn(x))

  # Print header
  cat("Tidyvec collection with", nrow(x), "items\n")
  cat("Embedding column:", emb_col, "\n")
  cat("Has embedding function:", ifelse(has_fn, "Yes", "No"), "\n")

  # Print the embedding dimension if available
  valid_embs <- x[[emb_col]][!vapply(x[[emb_col]], is.null, logical(1))]
  if (length(valid_embs) > 0) {
    cat("Embedding dimension:", length(valid_embs[[1]]), "\n")
  }

  # Print the table
  # Temporarily replace embedding column with a placeholder
  x_print <- x
  if (emb_col %in% names(x_print)) {
    x_print[[emb_col]] <- vapply(
      x_print[[emb_col]],
      function(v) if(is.null(v)) "<NULL>" else "<embedding>",
      character(1)
    )
  }

  # Call the next method
  NextMethod("print", x_print)

  invisible(x)
}

#' @export
`[.tidyvec` <- function(x, i, j, drop = FALSE) {
  out <- NextMethod()

  # Preserve tidyvec class and attributes only if we're subsetting rows
  if (missing(j)) {
    class(out) <- class(x)
    attrs <- attributes(x)
    # Remove dim, dimnames, row.names, names attributes
    attrs$dim <- NULL
    attrs$dimnames <- NULL
    attrs$names <- NULL
    attrs$row.names <- NULL
    for (a in names(attrs)) {
      attr(out, a) <- attrs[[a]]
    }
  }

  out
}

#' Compute embeddings for items in a tidyvec collection
#'
#' @param x A tidyvec object
#' @param content_column Column containing content to embed
#' @param embedding_fn Embedding function to use (overrides collection's function)
#' @param batch_size Number of items to process in each batch
#' @param force Whether to overwrite existing embeddings
#' @param ... Additional arguments passed to the embedding function
#' @return Updated tidyvec object with embeddings
#' @export
embed <- function(x, content_column, embedding_fn = NULL, batch_size = 50,
                  force = FALSE, ...) {
  if (!inherits(x, "tidyvec")) {
    stop("Not a tidyvec object")
  }

  # Get embedding column
  emb_col <- embedding_column(x)

  # Get embedding function
  fn <- embedding_fn %||% embedding_fn(x)
  if (is.null(fn)) {
    stop("No embedding function provided")
  }

  # Ensure content column exists
  if (missing(content_column) || !content_column %in% names(x)) {
    stop("Content column not found")
  }

  # Identify rows to process (missing or force)
  to_process <- rep(FALSE, nrow(x))
  for (i in seq_len(nrow(x))) {
    to_process[i] <- force || is.null(x[[emb_col]][[i]])
  }

  # If nothing to process, return early
  if (!any(to_process)) {
    return(x)
  }

  # Process in batches with progress bar
  indices <- which(to_process)
  n_batches <- ceiling(length(indices) / batch_size)

  if (requireNamespace("progress", quietly = TRUE)) {
    pb <- progress::progress_bar$new(
      format = "Computing embeddings [:bar] :percent eta: :eta",
      total = length(indices),
      clear = FALSE
    )
    pb$tick(0)
  } else {
    message("Computing embeddings for ", length(indices), " items...")
  }

  for (batch_idx in seq_len(n_batches)) {
    # Get batch indices
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, length(indices))
    batch_indices <- indices[start_idx:end_idx]

    # Extract content for this batch
    content <- x[[content_column]][batch_indices]

    # Compute embeddings (one at a time to avoid passing a list)
    for (j in seq_along(batch_indices)) {
      idx <- batch_indices[j]
      x[[emb_col]][[idx]] <- fn(content[[j]], ...)

      if (exists("pb")) pb$tick()
    }
  }

  if (!exists("pb")) {
    message("Done computing embeddings.")
  }

  x
}

#' Find nearest neighbors for a query in a tidyvec collection
#'
#' @param x A tidyvec object
#' @param query Query item (content or embedding)
#' @param n Number of results to return
#' @param as_embedding Whether the query is already an embedding vector
#' @param method Similarity method ("cosine", "euclidean", "dot")
#' @param min_score Minimum similarity score
#' @return Filtered tidyvec object with similarity scores
#' @export
nearest <- function(x, query, n = 5, as_embedding = FALSE,
                    method = c("cosine", "euclidean", "dot"),
                    min_score = 0) {
  if (!inherits(x, "tidyvec")) {
    stop("Not a tidyvec object")
  }

  # Get embedding column
  emb_col <- embedding_column(x)

  # Get query embedding
  if (!as_embedding) {
    fn <- embedding_fn(x)
    if (is.null(fn)) {
      stop("No embedding function available to process query")
    }
    query_embedding <- fn(query)
  } else {
    query_embedding <- query
  }

  if (is.null(query_embedding)) {
    stop("Could not create embedding for query")
  }

  # Choose similarity function
  method <- match.arg(method)
  sim_fn <- switch(method,
                   cosine = function(a, b) {
                     sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
                   },
                   euclidean = function(a, b) {
                     1 / (1 + sqrt(sum((a - b)^2)))
                   },
                   dot = function(a, b) {
                     sum(a * b)
                   }
  )

  # Filter out rows without embeddings
  has_embedding <- !vapply(x[[emb_col]], is.null, logical(1))
  if (sum(has_embedding) == 0) {
    return(x[0, ])
  }

  # Calculate similarities
  similarities <- numeric(nrow(x))
  for (i in seq_len(nrow(x))) {
    if (has_embedding[i]) {
      similarities[i] <- sim_fn(x[[emb_col]][[i]], query_embedding)
    }
  }

  # Add similarity column
  x$similarity <- similarities

  # Filter and sort
  result <- x[similarities >= min_score, ]
  result <- result[order(result$similarity, decreasing = TRUE), ]

  # Limit to n results
  if (nrow(result) > n) {
    result <- result[1:n, ]
  }

  result
}

# Add a helper to check collection status
#' Print details about a tidyvec collection
#'
#' @param x A tidyvec object
#' @return Invisibly returns the input object
#' @export
inspect_collection <- function(x) {
  if (!inherits(x, "tidyvec")) {
    cat("Not a tidyvec collection\n")
    return(invisible(x))
  }

  emb_col <- attr(x, "embedding_column", exact = TRUE) %||% "embedding"
  has_fn <- !is.null(attr(x, "embedding_fn", exact = TRUE))

  cat("Tidyvec collection with", nrow(x), "items\n")
  cat("Columns:", paste(names(x), collapse = ", "), "\n")
  cat("Embedding column:", emb_col, "\n")
  cat("Has embedding function:", ifelse(has_fn, "Yes", "No"), "\n")

  # Check embeddings
  if (emb_col %in% names(x)) {
    n_with_embeddings <- sum(!vapply(x[[emb_col]], is.null, logical(1)))
    cat("Items with embeddings:", n_with_embeddings, "/", nrow(x), "\n")

    if (n_with_embeddings > 0) {
      # Get first non-null embedding
      sample_emb <- NULL
      for (i in seq_along(x[[emb_col]])) {
        if (!is.null(x[[emb_col]][[i]])) {
          sample_emb <- x[[emb_col]][[i]]
          break
        }
      }

      if (!is.null(sample_emb)) {
        cat("Embedding dimension:", length(sample_emb), "\n")
      }
    }
  }

  invisible(x)
}
