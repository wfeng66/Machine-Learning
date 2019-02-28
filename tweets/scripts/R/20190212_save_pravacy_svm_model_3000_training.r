save_model <- function (){
    # The function should be run under the myscripts directory of server
    # Load training data from csv files
	d1 <- read.csv("twitter_data_processed.csv",header=F, stringsAsFactors=FALSE)
    d2 <- read.csv("result.csv",header=F, stringsAsFactors=FALSE)
	
	# Select balanced NOT Private and Private recodes
	colnames(d1)<- c("text","class")
	levels(d1$class) <- c(0,1)
	levels(d1$class) 
	d1 <- d1[order(d1$class),]  
	d1 <- rbind(d1[1:1100,],d1[4920:6024,])  

	# Randomly sort d1re
	set.seed(100)
	d1.row <- sample(1:nrow(d1),2204)
	d1 <- d1[d1.row,]

	d2 <- d2[,5:6]
	colnames(d2)<- c("text","class")
	levels(d2$class) <- c(0,1)

	# Combine Dataset that the first 2200 tweets from twitter_data_processed.csv and last 1100 tweets from result.csv
	data <- rbind(d1,d2[1:1100,])
	
	
	# Load packages
	library(RTextTools)
	source("text_classification_functions.R",local=T)  

	# Prepare the Dateset
	data.dtm.parameters <- c(language = "english",
                         removeNumbers = T,
                         removePunctuation = T,
                         removeSparseTerms = .999,
                         removeStopwords = T,
                         stemWords = T,
                         stripWhitespace = T,
                         toLower = T,
                         weighting = tm::weightTfIdf)


	data.corpus <- ProcessText(data$text)     
	data.dtm <- create_matrix(data.corpus,
                          language = data.dtm.parameters$language,
                          removeNumbers = data.dtm.parameters$removeNumbers,
                          removePunctuation = data.dtm.parameters$removePunctuation,
                          removeSparseTerms = data.dtm.parameters$removeSparseTerms,
                          removeStopwords = data.dtm.parameters$removeStopwords,
                          stemWords = data.dtm.parameters$stemWords,
                          stripWhitespace = data.dtm.parameters$stripWhitespace,
                          toLower = data.dtm.parameters$toLower,
                          weighting = data.dtm.parameters$weighting)
	
	data.index.train <- 1:3300
	data.index.test <- 3301:nrow(data)
	data.index <- list(training = data.index.train, testing = data.index.test)
					  
	data.container <- create_container(data.dtm,
                                   data$class,
                                   trainSize = data.index$training,
                                   testSize = data.index$testing,
                                   virgin = FALSE)
	
	svm.model <- train_model(data.container, "SVM")
	saveRDS(svm.model, file='privacy_svm_model.rds') 
		
	
}