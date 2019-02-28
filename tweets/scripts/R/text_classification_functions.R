# Install package dependencies
packages <- c("RTextTools", "tm")
#for (i in 1:length(packages)) if (!(packages[i] %in% installed.packages()[, 1])) install.packages(packages[i])
library(RTextTools)
library(tm)

# Temporary fix for the broken "create_matrix" function in RTextTools
create_matrix <- function (textColumns, language = "english", minDocFreq = 1, 
          maxDocFreq = Inf, minWordLength = 3, maxWordLength = Inf, 
          ngramLength = 1, originalMatrix = NULL, removeNumbers = FALSE, 
          removePunctuation = TRUE, removeSparseTerms = 0, removeStopwords = TRUE, 
          stemWords = FALSE, stripWhitespace = TRUE, toLower = TRUE, 
          weighting = weightTf) 
{
  stem_words <- function(x) {
    split <- strsplit(x, " ")
    return(wordStem(unlist(split), language = language))
  }
  tokenize_ngrams <- function(x, n = ngramLength) return(rownames(as.data.frame(unclass(textcnt(x, 
                                                                                                method = "string", n = n)))))
  control <- list(bounds = list(local = c(minDocFreq, maxDocFreq)), 
                  language = language, tolower = toLower, removeNumbers = removeNumbers, 
                  removePunctuation = removePunctuation, stopwords = removeStopwords, 
                  stripWhitespace = stripWhitespace, wordLengths = c(minWordLength, 
                                                                     maxWordLength), weighting = weighting)
  if (ngramLength > 1) {
    control <- append(control, list(tokenize = tokenize_ngrams), 
                      after = 7)
  }
  else {
    control <- append(control, list(tokenize = scan_tokenizer), 
                      after = 4)
  }
  if (stemWords == TRUE && ngramLength == 1) 
    control <- append(control, list(stemming = stem_words), 
                      after = 7)
  trainingColumn <- apply(as.matrix(textColumns), 1, paste, 
                          collapse = " ")
  trainingColumn <- sapply(as.vector(trainingColumn, mode = "character"), 
                           iconv, to = "UTF8", sub = "byte")
  corpus <- Corpus(VectorSource(trainingColumn), readerControl = list(language = language))
  matrix <- DocumentTermMatrix(corpus, control = control)
  if (removeSparseTerms > 0) 
    matrix <- removeSparseTerms(matrix, removeSparseTerms)
  if (!is.null(originalMatrix)) {
    terms <- colnames(originalMatrix[, which(!colnames(originalMatrix) %in% 
                                               colnames(matrix))])
    weight <- 0
    if (attr(weighting, "acronym") == "tf-idf") 
      weight <- 1e-09
    amat <- matrix(weight, nrow = nrow(matrix), ncol = length(terms))
    colnames(amat) <- terms
    rownames(amat) <- rownames(matrix)
    fixed <- as.DocumentTermMatrix(cbind(matrix[, which(colnames(matrix) %in% 
                                                          colnames(originalMatrix))], amat), weighting = weighting)
    matrix <- fixed
  }
  matrix <- matrix[, sort(colnames(matrix))]
  gc()
  return(matrix)
}

# Function to downsample datasets to prevent overfitting models
DownsampleData <- function(data, class.column = 2) {
  level.one <- levels(as.factor(data[, class.column]))[1]
  level.two <- levels(as.factor(data[, class.column]))[2]
  data.level.one <- data[data[, class.column] == level.one, ]
  data.level.two <- data[data[, class.column] == level.two, ]
  nrow.data.level.one <- nrow(data.level.one)
  nrow.data.level.two <- nrow(data.level.two)
  if (nrow.data.level.one > nrow.data.level.two) {
    data.level.one <- data.level.one[sample(nrow.data.level.two), ]
    data <- rbind(data.level.one, data.level.two)
    return(data[sample(nrow(data)), ])
  } else if (num.data.level.one < num.data.level.two) {
    data.level.two <- data.level.two[sample(nrow.data.level.one), ]
    data <- rbind(data.level.one, data.level.two)
    return(data[sample(nrow(data)), ])
  } else { return(data) }
}

# Function to generate an index of a randomly-selected portion of a 
# dataset to be used to subset the original dataset into training and
# testing datasets
PartitionData <- function(data, proportion = 0.66667) {
  #data <- data[sample(nrow(data)), ]
  data.index.train <- 1:(nrow(data) * proportion)
  data.index.test <- (nrow(data) * proportion + 1):nrow(data)
  list(training = data.index.train, testing = data.index.test)
}

# Convert XML characters to their ASCII equivalent
XMLToASCII <- function(x) {
  x <- gsub("&amp;", "&", x)
  x <- gsub("&quot;", "\"", x)
  x <- gsub("&#039;", "'", x)
  x <- gsub("&lt;", "<", x)
  x <- gsub("&gt;", ">", x)
}

# Function to make text ASCII compliant
MakeASCII <- function(x) {
  require(stringi)
  x <- XMLToASCII(x)
  x <- stringi::stri_trans_general(x, "Latin-ASCII")
  x <- gsub("�", "", x)
  return(x)
}

# Remove Retweet indicator from text
RemoveRetweetInfo <- function(x) gsub("(RT|via)((?:\\b\\W*@\\w+)+)(:[[:space:]])", "", x)

# Function to collapse contractions before punctuation removal
CollapseContractions <- function (x) {
  x <- gsub("ain't", "isnt", x)
  x <- gsub("won't", "wont", x)
  x <- gsub("cannot", "cant", x)
  x <- gsub("can't", "cant", x)
  x <- gsub("(?<=[[:alpha:]])n' ", "ng ", x, perl = T)
  x <- gsub("(?<=[[:alpha:]])n't", "nt", x, perl = T)
  x <- gsub("(?<=[[:alpha:]])'d", "d", x, perl = T)
  x <- gsub("(?<=[[:alpha:]])'ll", "ll", x, perl = T)
  x <- gsub("(?<=[[:alpha:]])'m", "m", x, perl = T)
  x <- gsub("(?<=[[:alpha:]])'s", "s", x, perl = T)
  x <- gsub("(?<=[[:alpha:]])'ve", "ve", x, perl = T)
  return(x)
}

# N-gram tokenizer
NGramTokenizer <- function(x, n = 2) {
  trim <- function(x) gsub("(^\\s+|\\s+$)", "", x)
  terms <- strsplit(trim(x), split = "\\s+")[[1]]
  ngrams <- vector()
  if (length(terms) >= n) {
    for (i in n:length(terms)) {
      ngram <- paste(terms[(i-n+1):i], collapse = " ")
      ngrams <- c(ngrams, ngram)
    }
  }
  ngrams  
}

# Function to stem words (to compensate for RTextTools' broken stemming)
StemWords <- function(x) {
  x <- lapply(x, NGramTokenizer, n = 1)
  for (i in length(x)) {
    if (length(x[[i]] < 255)) {
      wordStem(x[[i]])
    } else {
      x[[-i]]
    }
  }
  x <- lapply(x, paste, collapse = " ")
}

# Function to process text before classification
ProcessText <- function(text) {
  text <- RemoveRetweetInfo(text)
  text <- MakeASCII(text)
  text <- tolower(text)
  text <- CollapseContractions(text)
  text <- StemWords(text)
  return(text)
}

# Function to classify text if you have a model package item containing the model and parameters
ClassifyText <- function(text, model.package, tweets = F) {
  text.marker <- length(text)
  if (text.marker == 1) text <- list(text, "Lorem ipsum")
  text <- ProcessText(text)
  text.dtm <- create_matrix(text,
                            language = model.package[[2]]$language,
                            originalMatrix = model.package[[1]],
                            removeNumbers = model.package[[2]]$removeNumbers,
                            removePunctuation = model.package[[2]]$removePunctuation,
                            removeSparseTerms = model.package[[2]]$removeSparseTerms, 
                            removeStopwords = model.package[[2]]$removeStopwords,
                            stemWords = model.package[[2]]$stemWords,
                            stripWhitespace = model.package[[2]]$stripWhitespace,
                            toLower = model.package[[2]]$toLower, 
                            weighting = model.package[[2]]$weighting)
  text.container <- RTextTools::create_container(text.dtm,
                                                 labels = NULL,
                                                 testSize = 1:length(text),
                                                 virgin = TRUE)
  text.result <- classify_model(text.container, model.package[[3]])
  if (text.marker == 1) {
    text <- text[-2]
    text.result <- text.result[-2, ]
  }
  return(text.result[, 1])
}

# Function to create results table of a model comparing predicted values with actual values
CreateModelResultsTable <- function(model.results.class, data.class) table(Predicted = model.results.class, Actual = data.class)

# Function to return the accuracy of a model
MeasureModelAccuracy <- function(model.results.table) {
  model.results.table.correct <- model.results.table[1, 1] + model.results.table[2, 2]
  return(paste0(((model.results.table.correct / (model.results.table.correct + model.results.table[1, 2] + model.results.table[2, 1])) * 100), "%"))
}