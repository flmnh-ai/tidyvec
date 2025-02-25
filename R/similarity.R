#' Similarity operator
#'
#' @param a First vector or tidyvec object
#' @param b Second vector or tidyvec object
#' @param method Similarity method to use ("cosine", "euclidean", or "dot")
#' @return Similarity score or tidyvec object with similarity scores
#' @export
`%~%` <- function(a, b, method = "cosine") {
  # If both are vectors, calculate similarity directly
  if (is.numeric(a) && is.numeric(b)) {
    # Basic vector similarity
    switch(method,
           cosine = sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2))),
           euclidean = 1 / (1 + sqrt(sum((a - b)^2))),
           dot = sum(a * b)
    )
  } else if (inherits(a, "tidyvec") && (is.numeric(b) || is.character(b))) {
    # tidyvec-to-query similarity
    nearest(a, b, n = nrow(a), method = method)
  } else if (inherits(b, "tidyvec") && (is.numeric(a) || is.character(a))) {
    # query-to-tidyvec similarity
    nearest(b, a, n = nrow(b), method = method)
  } else {
    stop("Unsupported operand types for %~%")
  }
}
