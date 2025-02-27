% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/embedders.R
\name{embedder_tfidf}
\alias{embedder_tfidf}
\title{Create a simple TF-IDF embedding function for text}
\usage{
embedder_tfidf(corpus, min_freq = 2)
}
\arguments{
\item{corpus}{Text corpus to build vocabulary}

\item{min_freq}{Minimum term frequency}
}
\value{
An embedding function
}
\description{
Create a simple TF-IDF embedding function for text
}
