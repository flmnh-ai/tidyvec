#' @importFrom rlang .data
NULL
utils::globalVariables(c("x", "y"))

#' Visualize embedding space using dimensionality reduction
#'
#' @param x A tidyvec object
#' @param method Dimensionality reduction method ("tsne", "umap", "pca")
#' @param labels Column to use for point labels
#' @param color Column to use for point colors
#' @param n_neighbors Number of neighbors (for UMAP)
#' @param perplexity Perplexity parameter (for t-SNE)
#' @param images_column Optional column containing image paths to use instead of points
#' @param ... Additional arguments passed to the plotting function
#' @return A ggplot object
#' @export
viz_embeddings <- function(x, method = c("umap", "tsne", "pca"),
                           labels = NULL, color = NULL,
                           n_neighbors = 15, perplexity = 30, images_column = NULL, ...) {
  if (!inherits(x, "tidyvec")) {
    stop("Not a tidyvec object")
  }

  # Get embedding column
  emb_col <- embedding_column(x)

  # Filter to rows with embeddings
  has_embedding <- !vapply(x[[emb_col]], is.null, logical(1))
  x_valid <- x[has_embedding, ]

  if (nrow(x_valid) == 0) {
    stop("No valid embeddings found")
  }

  # Extract embeddings matrix
  embeddings <- do.call(rbind, x_valid[[emb_col]])

  # Apply dimensionality reduction
  method <- match.arg(method)
  coords <- switch(method,
                   umap = {
                     if (!requireNamespace("umap", quietly = TRUE)) {
                       stop("Package 'umap' required for UMAP visualization")
                     }
                     set.seed(42)  # For reproducibility
                     umap_result <- umap::umap(embeddings, n_neighbors = n_neighbors)
                     umap_result$layout
                   },
                   tsne = {
                     if (!requireNamespace("Rtsne", quietly = TRUE)) {
                       stop("Package 'Rtsne' required for t-SNE visualization")
                     }
                     set.seed(42)  # For reproducibility
                     tsne_result <- Rtsne::Rtsne(embeddings, perplexity = perplexity,
                                                 check_duplicates = FALSE)
                     tsne_result$Y
                   },
                   pca = {
                     pca_result <- stats::prcomp(embeddings, scale. = TRUE)
                     pca_result$x[, 1:2]
                   }
  )

  # Create plot data
  plot_data <- x_valid
  plot_data$x <- coords[, 1]
  plot_data$y <- coords[, 2]

  # Create plot with ggplot2
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required for visualization")
  }

  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = x, y = y))

  # Add colors if specified
  if (!is.null(color) && color %in% names(plot_data)) {
    p <- p + ggplot2::aes(color = .data[[color]])
  }

  # Add points
  p <- p + ggplot2::geom_point(alpha = 0.7, size = 3)

  # Add labels if specified
  if (!is.null(labels) && labels %in% names(plot_data)) {
    p <- p + ggplot2::geom_text(
      ggplot2::aes(label = .data[[labels]]),
      hjust = -0.1,
      vjust = 0.1,
      size = 3
    )
  }

  # Add theme and title
  p <- p +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = paste("Embedding Space Visualization using", toupper(method)),
      x = paste(method, "1"),
      y = paste(method, "2")
    )

  # optionally display images
  if (!is.null(images_column) && images_column %in% names(plot_data)) {
    if (!requireNamespace("ggimage", quietly = TRUE)) {
      stop("Package 'ggimage' is required for image embedding visualization")
    }
    p <- p + ggimage::geom_image(ggplot2::aes(image = .data[[images_column]]),
                                 size = 0.05)
  }

  p
}

#' Visualize images in a tidyvec collection using magick
#'
#' @param x A tidyvec object containing images
#' @param path_column Column containing image paths or URLs
#' @param n Number of images to display (default: all)
#' @param ncol Number of columns in the grid (default: 3)
#' @param width Width of each image in pixels (default: 200)
#' @param include_similarity Whether to show similarity scores if available (default: TRUE)
#' @param label_columns Additional columns to use as labels
#' @return A magick image object
#' @export
viz_images <- function(x,
                       path_column,
                       n = NULL,
                       ncol = 3,
                       width = 200,
                       include_similarity = TRUE,
                       label_columns = NULL) {
  # Check for required packages
  if (!requireNamespace("magick", quietly = TRUE)) {
    stop("Package 'magick' is required. Please install with: install.packages('magick')")
  }

  # Ensure path_column exists
  if (!path_column %in% names(x)) {
    stop("Path column '", path_column, "' not found in data")
  }

  # Limit number of images if specified
  if (!is.null(n) && n < nrow(x)) {
    x <- x[1:n, ]
  }

  # Calculate grid dimensions
  nrow_grid <- ceiling(nrow(x) / ncol)

  # Load and prepare images
  images <- list()
  for (i in 1:nrow(x)) {
    # Create label
    label_parts <- c()

    # Add similarity if requested and available
    if (include_similarity && "similarity" %in% names(x)) {
      label_parts <- c(label_parts,
                       sprintf("Similarity: %.3f", x$similarity[i]))
    }

    # Add additional labels if requested
    if (!is.null(label_columns)) {
      for (col in label_columns) {
        if (col %in% names(x)) {
          label_parts <- c(label_parts,
                           sprintf("%s: %s", col, x[[col]][i]))
        }
      }
    }

    label <- if (length(label_parts) > 0) paste(label_parts, collapse = "\n") else NULL

    # Read image
    img <- magick::image_read(x[[path_column]][i])

    # Resize to maintain aspect ratio within width
    img <- magick::image_scale(img, paste0(width, "x"))

    # Add label if available
    if (!is.null(label)) {
      # Create a label image
      label_img <- magick::image_blank(width, 50, "white")
      label_img <- magick::image_annotate(
        label_img,
        label,
        size = 10,
        gravity = "northwest",
        color = "black"
      )

      # Stack image and label
      img <- magick::image_append(
        c(img, label_img),
        stack = TRUE
      )
    }

    images[[i]] <- img
  }

  # Arrange images in a grid
  grid_rows <- list()
  for (row in 1:nrow_grid) {
    start_idx <- (row - 1) * ncol + 1
    end_idx <- min(row * ncol, length(images))

    if (start_idx <= length(images)) {
      # Create this row
      row_images <- images[start_idx:end_idx]

      # Pad with blank images if needed to complete the row
      if (length(row_images) < ncol) {
        for (i in (length(row_images) + 1):ncol) {
          row_images[[i]] <- magick::image_blank(width, width, "white")
        }
      }

      # Append images horizontally for this row
      grid_rows[[row]] <- magick::image_append(do.call(c, row_images))
    }
  }

  # Append rows vertically to create the final grid
  if (length(grid_rows) > 0) {
    grid <- magick::image_append(do.call(c, grid_rows), stack = TRUE)

    # Add border for better visibility
    grid <- magick::image_border(grid, "lightgray", "2x2")

    # Print and return
    return(print(grid, info = FALSE))
  } else {
    warning("No images to display")
    invisible(NULL)
  }
}
