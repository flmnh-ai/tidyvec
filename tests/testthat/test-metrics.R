test_that("ap_at_k calculates correct average precision", {
  # Perfect retrieval
  relevant <- c("A", "B", "C")
  retrieved <- c("A", "B", "C", "D", "E")
  expect_equal(ap_at_k(relevant, retrieved, k = 5), 1.0)

  # All relevant items found, but not in top positions
  retrieved2 <- c("D", "E", "A", "B", "C")
  ap <- ap_at_k(relevant, retrieved2, k = 5)
  expect_gt(ap, 0)
  expect_lt(ap, 1)

  # No relevant items found
  retrieved3 <- c("D", "E", "F", "G", "H")
  expect_equal(ap_at_k(relevant, retrieved3, k = 5), 0)

  # Partial retrieval
  relevant4 <- c("A", "B", "C", "D")
  retrieved4 <- c("A", "X", "B", "Y", "Z")
  # Precision at positions: 1/1 at pos 1, 2/3 at pos 3
  # AP = (1.0 + 0.667) / 2 = 0.833
  expect_equal(ap_at_k(relevant4, retrieved4, k = 5), (1.0 + 2/3) / 2, tolerance = 0.01)
})

test_that("ap_at_k handles edge cases", {
  # Empty relevant
  expect_equal(ap_at_k(character(0), c("A", "B"), k = 5), 0)

  # k larger than retrieved
  relevant <- c("A", "B")
  retrieved <- c("A", "B")
  expect_equal(ap_at_k(relevant, retrieved, k = 10), 1.0)

  # Single item
  expect_equal(ap_at_k("A", c("A", "B"), k = 1), 1.0)
  expect_equal(ap_at_k("A", c("B", "A"), k = 1), 0.0)
})

test_that("ap_at_k works with numeric IDs", {
  # Perfect retrieval with numeric IDs
  relevant <- c(1, 2, 3)
  retrieved <- c(1, 2, 3, 4, 5)
  expect_equal(ap_at_k(relevant, retrieved, k = 5), 1.0)

  # Partial retrieval with numeric IDs
  relevant2 <- c(1, 2, 3, 4)
  retrieved2 <- c(1, 10, 2, 20, 30)
  # Precision at positions: 1/1 at pos 1, 2/3 at pos 3
  expect_equal(ap_at_k(relevant2, retrieved2, k = 5), (1.0 + 2/3) / 2, tolerance = 0.01)

  # No relevant items found with numeric IDs
  expect_equal(ap_at_k(c(1, 2, 3), c(4, 5, 6), k = 3), 0)
})

test_that("ap_at_k validates inputs", {
  expect_error(ap_at_k(list(1, 2), c("A", "B"), k = 5))  # List not allowed
  expect_error(ap_at_k(c("A"), c("B"), k = 0))
  expect_error(ap_at_k(c("A"), c("B"), k = -1))
})

test_that("precision_at_k calculates correct precision", {
  relevant <- c("A", "B", "C")

  # All top-k are relevant
  retrieved1 <- c("A", "B", "C", "D", "E")
  expect_equal(precision_at_k(relevant, retrieved1, k = 3), 1.0)

  # Half are relevant
  retrieved2 <- c("A", "D", "B", "E", "C")
  expect_equal(precision_at_k(relevant, retrieved2, k = 4), 0.5)

  # None are relevant
  retrieved3 <- c("D", "E", "F")
  expect_equal(precision_at_k(relevant, retrieved3, k = 3), 0.0)
})

test_that("precision_at_k works with numeric IDs", {
  relevant <- c(1, 2, 3)

  # All top-k are relevant
  expect_equal(precision_at_k(relevant, c(1, 2, 3, 4, 5), k = 3), 1.0)

  # Half are relevant
  expect_equal(precision_at_k(relevant, c(1, 10, 2, 20, 3), k = 4), 0.5)

  # None are relevant
  expect_equal(precision_at_k(relevant, c(10, 20, 30), k = 3), 0.0)
})

test_that("recall_at_k calculates correct recall", {
  relevant <- c("A", "B", "C", "D")

  # All relevant items found
  retrieved1 <- c("A", "B", "C", "D", "E")
  expect_equal(recall_at_k(relevant, retrieved1, k = 5), 1.0)

  # Half found
  retrieved2 <- c("A", "B", "X", "Y", "Z")
  expect_equal(recall_at_k(relevant, retrieved2, k = 5), 0.5)

  # None found
  retrieved3 <- c("X", "Y", "Z")
  expect_equal(recall_at_k(relevant, retrieved3, k = 3), 0.0)
})

test_that("recall_at_k handles empty relevant set", {
  expect_equal(recall_at_k(character(0), c("A", "B"), k = 5), 0)
})

test_that("recall_at_k works with numeric IDs", {
  relevant <- c(1, 2, 3, 4)

  # All relevant items found
  expect_equal(recall_at_k(relevant, c(1, 2, 3, 4, 5), k = 5), 1.0)

  # Half found
  expect_equal(recall_at_k(relevant, c(1, 2, 10, 20, 30), k = 5), 0.5)

  # None found
  expect_equal(recall_at_k(relevant, c(10, 20, 30), k = 3), 0.0)
})

test_that("calculate_baseline computes correct baseline", {
  library(dplyr)
  library(tibble)

  # Uniform distribution: 3 categories with equal frequency
  # p_A = p_B = p_C = 1/3
  # baseline = 3 * (1/3)^2 = 1/3
  data <- tibble(
    id = 1:9,
    category = rep(c("A", "B", "C"), each = 3)
  )

  expect_equal(calculate_baseline(data, "category"), 1/3, tolerance = 0.01)

  # Skewed distribution: 80% A, 20% B
  # baseline = 0.8^2 + 0.2^2 = 0.64 + 0.04 = 0.68
  data2 <- tibble(
    id = 1:10,
    category = c(rep("A", 8), rep("B", 2))
  )

  expect_equal(calculate_baseline(data2, "category"), 0.68, tolerance = 0.01)
})

test_that("calculate_baseline handles missing values", {
  library(dplyr)
  library(tibble)

  data <- tibble(
    id = 1:10,
    category = c("A", "A", "B", "B", NA, NA, "null", "null", "A", "B")
  )

  # Should ignore NA and "null" values
  # 3 A's, 3 B's -> baseline = 2 * (0.5)^2 = 0.5
  expect_equal(calculate_baseline(data, "category"), 0.5, tolerance = 0.01)
})

test_that("calculate_map works with tidyvec objects", {
  library(tibble)
  library(dplyr)

  # Create simple test collection
  data <- tibble(
    id = as.character(1:10),
    category = rep(c("A", "B"), each = 5),
    embedding = lapply(1:10, function(i) {
      # Create embeddings where items in same category are similar
      base <- if (i <= 5) c(1, 0) else c(0, 1)
      base + rnorm(2, 0, 0.1)
    })
  )

  collection <- vec(data)

  # Calculate MAP
  map_score <- calculate_map(collection, "category", k = 5, progress = FALSE)

  # Should be better than random (0.5)
  expect_gt(map_score, 0.5)
  expect_lte(map_score, 1.0)
})

test_that("calculate_map validates inputs", {
  library(tibble)

  data <- tibble(id = 1:5, cat = c("A", "B", "A", "B", "A"))
  collection <- vec(data)

  expect_error(calculate_map(data, "cat"))  # Not a tidyvec object
  expect_error(calculate_map(collection, "nonexistent"))  # Variable doesn't exist
  expect_error(calculate_map(collection, "cat", id_column = "missing"))  # Bad ID column
})

test_that("query_metrics returns all metrics", {
  relevant <- c("A", "B", "C")
  retrieved <- c("A", "X", "B", "Y", "C")

  metrics <- query_metrics(relevant, retrieved, k = 5)

  expect_named(metrics, c("ap", "precision", "recall"))
  expect_type(metrics$ap, "double")
  expect_type(metrics$precision, "double")
  expect_type(metrics$recall, "double")

  # All should be between 0 and 1
  expect_gte(metrics$ap, 0)
  expect_lte(metrics$ap, 1)
  expect_gte(metrics$precision, 0)
  expect_lte(metrics$precision, 1)
  expect_gte(metrics$recall, 0)
  expect_lte(metrics$recall, 1)
})

test_that("evaluate_retrieval works with multiple variables", {
  library(tibble)
  library(dplyr)

  # Create test collection with multiple attributes
  data <- tibble(
    id = as.character(1:20),
    cat1 = rep(c("A", "B"), each = 10),
    cat2 = rep(c("X", "Y", "Z"), length.out = 20),
    embedding = lapply(1:20, function(i) rnorm(3))
  )

  collection <- vec(data)

  results <- evaluate_retrieval(
    collection,
    variables = c("cat1", "cat2"),
    k = 5,
    progress = FALSE
  )

  expect_s3_class(results, "tbl_df")
  expect_named(results, c("variable", "map", "baseline", "improvement"))
  expect_equal(nrow(results), 2)
  expect_true(all(results$variable %in% c("cat1", "cat2")))
})
