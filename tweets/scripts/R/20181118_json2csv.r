json2csv <- function(pa){
 path<-"H:/CSI/Honor Program/Twitters/Data"
 files <- dir(pa)
 numOfiles=length(files)
 for(i in 1:numOfiles){
 json_file<- paste(pa,files[i],sep='/')
 fileName <- unlist(strsplit(files[i],"[.]"))[1]
 csv_file<- paste(pa,paste(fileName,"csv",sep='.'),sep='/')
 dat <- fromJSON(sprintf("[%s]", paste(readLines(json_file), collapse=",")))
 dt<- dat[c(1:5,13)]
 write.csv(dt, csv_file)}}
 
 
 # syntax: json2csv(path)
 # example: install.packages("jsonlite")
 #          library(jsonlite)
 #			path="c:/Data"
 #			json2csv(path)
