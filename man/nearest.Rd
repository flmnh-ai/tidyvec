% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/core.R
\name{nearest}
\alias{nearest}
\title{Find nearest neighbors for a query in a tidyvec collection}
\usage{
nearest(
  x,
  query,
  n = 5,
  as_embedding = FALSE,
  method = c("cosine", "euclidean", "dot"),
  min_score = 0
)
}
\arguments{
\item{x}{A tidyvec object}

\item{query}{Query item (content or embedding)}

\item{n}{Number of results to return}

\item{as_embedding}{Whether the query is already an embedding vector}

\item{method}{Similarity method ("cosine", "euclidean", "dot")}

\item{min_score}{Minimum similarity score}
}
\value{
Filtered tidyvec object with similarity scores
}
\description{
Find nearest neighbors for a query in a tidyvec collection
}
