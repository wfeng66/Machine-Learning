# This function is used for reusing the svm model in flexible training size
# The first parameter is training data set size, the second one is the model file name
# For use training size of 200 tweets and file name as privacy_svm_model.rds, it should be:
# reuse_model(200, "privacy_svm_model.rds")
reuse_model <- function (training_size, model_file){
    # The function should be run under the myscripts directory of server
    # Load training data from csv files
	d1 <- read.csv("twitter_data_processed.csv",header=F, stringsAsFactors=FALSE)
    d2 <- read.csv("result.csv",header=F, stringsAsFactors=FALSE)
	
	# Select balanced NOT Private and Private recodes
	colnames(d1)<- c("text","class")
	levels(d1$class) <- c(0,1)
	levels(d1$class) 
	d1 <- d1[order(d1$class),]  
	d1 <- rbind(d1[1:(training_size/2),],d1[(6024-(training_size/2)):6024,])  

	# Randomly sort d1re
	set.seed(100)
	d1.row <- sample(1:nrow(d1),training_size)
	data <- d1[d1.row,]


	
	# Retrieve predicted data from database 
	# Load the PSQL library
	library(RPostgreSQL)

	# establish connection with database
	con <- dbConnect(
		drv = dbDriver("PostgreSQL")
		, host = "localhost"
		, port = 5234
		, dbname = "privacy"
		, user = "privacy_admin"
		, password = "privacy5234"
	)

	# Create the query which retrieves tweets have been tagged
	query_statement <- "SELECT text, isprivate FROM tweets WHERE isprivate IS NOT NULL;"
	result_set <- dbSendQuery(con, query_statement)

	# set the number of rows to fetch 
	number_of_rows_to_fetch <- 10000

	# retrieves the data
	df <- fetch(result_set, number_of_rows_to_fetch)
	colnames(df)<- c("text","class")
	df$class[df$class==TRUE]<- "Private"
	df$class[df$class==FALSE]<- "Not Private"
	
	# Combine training data set and predicted data set 
	data <- rbind(data,df)

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
	
	data.index.train <- 1:training_size
	data.index.test <- (training_size+1):nrow(data)
	data.index <- list(training = data.index.train, testing = data.index.test)
					  
	data.container <- create_container(data.dtm,
                                   data$class,
                                   trainSize = data.index$training,
                                   testSize = data.index$testing,
                                   virgin = FALSE)	
								   
	# Load the saved model
	mymodel <- readRDS(model_file) 
    
    # Predict	
    svm.result <- classify_model(data.container, mymodel)
	
	# verify the result
	head(svm.result)
	svm.table <- CreateModelResultsTable(svm.result[, 1], data[data.index$testing, ]$class)
	svm.table
	svm.accuracy <- MeasureModelAccuracy(svm.table)
	svm.accuracy
	
}