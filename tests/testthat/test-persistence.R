test_that("write_vec strips embedding function", {
  # Create a simple embedder
  embedder <- function(x) list(c(1, 2, 3))

  # Create collection
  df <- tibble::tibble(id = 1, text = "test")
  collection <- vec(df, embedding_fn = embedder)

  # Verify embedding function is present
  expect_false(is.null(attr(collection, "embedding_fn")))

  # Save to temp file
  tmpfile <- tempfile(fileext = ".qs")
  write_vec(collection, tmpfile)

  # Load back
  loaded <- qs::qread(tmpfile)

  # Verify embedding function was stripped
  expect_null(attr(loaded, "embedding_fn"))

  # Cleanup
  unlink(tmpfile)
})

test_that("read_vec can restore embedding function", {
  # Create a simple embedder
  embedder <- function(x) list(c(1, 2, 3))

  # Create and save collection
  df <- tibble::tibble(id = 1, text = "test")
  collection <- vec(df, embedding_fn = embedder)

  tmpfile <- tempfile(fileext = ".qs")
  write_vec(collection, tmpfile)

  # Load without embedder
  loaded1 <- read_vec(tmpfile)
  expect_null(attr(loaded1, "embedding_fn"))

  # Load with embedder
  new_embedder <- function(x) list(c(4, 5, 6))
  loaded2 <- read_vec(tmpfile, embedding_fn = new_embedder)
  expect_false(is.null(attr(loaded2, "embedding_fn")))

  # Verify it's the new embedder
  expect_equal(attr(loaded2, "embedding_fn")(NULL), list(c(4, 5, 6)))

  # Cleanup
  unlink(tmpfile)
})

test_that("nearest gives helpful error without embedding function", {
  # Create collection with embeddings but no function
  df <- tibble::tibble(
    id = 1:3,
    text = c("a", "b", "c"),
    embedding = list(c(1, 0), c(0, 1), c(1, 1))
  )
  collection <- vec(df)  # No embedding function

  # Should error with helpful message
  expect_error(
    nearest(collection, "new query"),
    "read_vec\\(file, embedding_fn"
  )

  # Should work with as_embedding = TRUE
  result <- nearest(collection, c(1, 0), as_embedding = TRUE, n = 1)
  expect_equal(nrow(result), 1)
})

test_that("embed gives helpful error without embedding function", {
  # Create collection without function
  df <- tibble::tibble(id = 1, text = "test")
  collection <- vec(df)

  # Should error with helpful message
  expect_error(
    embed(collection, "text"),
    "read_vec\\(file, embedding_fn"
  )
})
