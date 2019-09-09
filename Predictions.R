# Load packages, datasets and models ####
pacman::p_load("readr", "caret", "ggplot2", "MASS", "reshape", "reshape2", "crunch", "dplyr", "lattice",
               "randomForest", "ranger", "anchors", "FNN")



training_data<- read.csv("trainingData.csv", sep = ",")
validation_data<- read.csv("testData.csv", sep = ",")

x<- readRDS("Model_x")
y<- readRDS("Model_y")
z<- readRDS("Model_z")
base_randomforest<- readRDS("Model_base_randomforest")
random_forest_longitude<- readRDS("Model_random_forest_longitude")
random_forest_latitude<- readRDS("Model_random_forest_latitude")
randomForestBuilding<- readRDS("Model_random_forest_building")
cascadingRandomForestLongitude<- readRDS("Model_cascading_random_forest_longitude")
cascadingRandonForestLatitude<- readRDS("Model_cascadingRandomForestLatitude")
cascadingRandomForestFloor<- readRDS("Model_cascadingRandomForestFloor")
latitudeKNN<- readRDS("LatitudeKNN")
longitudeKNN<- readRDS("LongitudeKNN")
floorKNN<- readRDS("FloorKNN")

# Preprocessing of the data ####

is.na(training_data)
str(training_data)
summary(training_data)
training_data<- as.data.frame(training_data)

# Making a plot of the distribution of RSSI values
meltedWAPs<- melt(training_data, id = c(521:529))
str(meltedWAPs)
hist(meltedWAPs$value,
     main = "Distribution of WAPs",
     xlim = c(-110,0), 
     ylim = c(0,120000),
     breaks = 30,
     xlab = "RSSI")

# Remove redundant columns without information
noVarianceCols<- nearZeroVar(training_data, uniqueCut = 0) 
trimmedSet<- training_data[, -c(noVarianceCols)]



class(training_data$LATITUDE)

# Checking which variable is easiest to predict

trimmedSet$LATITUDE <- as.numeric(trimmedSet$LATITUDE)
#x <-train(LATITUDE~., data = trimmedSet[,c(1:465, 467)], method = "lm")
summary(x)
postResample(x$finalModel$fitted.values, trimmedSet$LATITUDE)
# Rsquared of 92.4
#model_x<- saveRDS(x, file = "Model_x")

trimmedSet$LONGITUDE<- as.numeric(trimmedSet$LONGITUDE)
#y<- train(LONGITUDE~., data = trimmedSet[, c(1:466)], method = "lm")
summary(y)
postResample(y$finalModel$fitted.values, trimmedSet$LONGITUDE)
#Rsquared of 94.38
#model_y<- saveRDS(y, file = "Model_y")

trimmedSet$FLOOR<- as.factor(trimmedSet$FLOOR)
#z<- randomForest::randomForest(FLOOR~ .,trimmedSet[, c(1:465, 468)], ntree = 10)
z
# Accuracy of 94.14
#model_z<- saveRDS(z, file = "Model_z")
# It seems that the floor is the hardest variable to predict.

# Building a base model ####

# Predicting floor with random forest, as it seems the easiest to predict
toModel<- trimmedSet[, c(1:469)]
set.seed(123)

data_partition<- createDataPartition(y = toModel$FLOOR, p = 0.7, list = FALSE)
training_set<- toModel[data_partition, ]
testing_set<- toModel[-data_partition, ]

WAP <- grep(x = names(training_set), pattern = "WAP")
train_mod<- training_set[WAP]


# base_randomforest<- randomForest(y = training_set$FLOOR, x = train_mod , ntree = 150)
#saveRDS(base_randomforest, file = "Model_base_randomforest")

base_randomforest
plot(base_randomforest)


predicting_floor_on_testset<- predict(base_randomforest, newdata = testing_set)
confusionMatrix(predicting_floor_on_testset, testing_set$FLOOR)
# Accuracy 99,11%

training_set$predictedFloor<- base_randomforest$predicted
validation_data$FLOOR<- as.factor(validation_data$FLOOR)

predicting_floor_on_validationset<- predict(base_randomforest, newdata = validation_data[, c(1:520, 523)])
confusionMatrix(predicting_floor_on_validationset, validation_data$FLOOR)
# Accuracy on the validation set of 86.6%, Kappa = 0.8105
output1<- c()
output1$FLOOR<- predicting_floor_on_validationset
output1<- as.data.frame(output1)

# Predicting longitude 
training_set$LONGITUDE<- as.numeric(training_set$LONGITUDE)

#random_forest_longitude<- randomForest(y = training_set$LONGITUDE, x = train_mod, ntree = 150)
random_forest_longitude
#saveRDS(random_forest_longitude, file = "Model_random_forest_longitude")
plot(random_forest_longitude)
summary(random_forest_longitude)


predicting_longitude_on_testset<- predict(random_forest_longitude, newdata = testing_set[, c(1:465)])
predicting_longitude_on_testset
postResample(testing_set$LONGITUDE, predicting_longitude_on_testset)
# RMSE = 8.4, R2 = 0.9954, MAE = 3.81

predicting_longitude_on_validationset<- predict(random_forest_longitude, newdata = validation_data[, c(1:521)])
postResample(predicting_longitude_on_validationset, validation_data$LONGITUDE)
# RMSE = 13.6, R2 = 0.9876, MAE = 9.11
output1$LONGITUDE<- predicting_longitude_on_validationset
# Predicting latitude
training_set$LATITUDE<- as.numeric(training_set$LATITUDE)
#random_forest_latitude<- randomForest(y = training_set$LATITUDE, x = train_mod, ntree = 150)
random_forest_latitude
#saveRDS(random_forest_latitude, file = "Model_random_forest_latitude")
plot(random_forest_latitude)
summary(random_forest_latitude)

predicting_latitude_on_testset<- predict(random_forest_latitude, newdata = testing_set[, c(1:465)])
predicting_latitude_on_testset
postResample(testing_set$LATITUDE, predicting_latitude_on_testset)
# RMSE = 5.8988, R2 = 0.9923, MAE = 2.937

predicting_latitude_on_validationset<- predict(random_forest_latitude, newdata = validation_data[, c(1:520, 522)])
postResample(validation_data$LATITUDE, predicting_latitude_on_validationset)
# RMSE = 11.52, R2 = 0.9749, MAE = 7.64

output1$LATITUDE<- predicting_latitude_on_validationset
output1<- output1[c("LATITUDE", "LONGITUDE", "FLOOR")]
write.csv(output1, file = "Model1Predictions.csv")
# Error analysis
validation_data$FLOOR<- as.character(validation_data$FLOOR)
validation_data$FLOOR<- as.numeric(validation_data$FLOOR)
predicting_floor_on_validationset<- as.numeric(predicting_floor_on_validationset)

errorDataFrame<- data.frame(Floor = validation_data$FLOOR,
                            FloorPred = predicting_floor_on_validationset,
                            Longitude = validation_data$LONGITUDE,
                            LongitudePred = predicting_longitude_on_validationset,
                            Latitide = validation_data$LATITUDE,
                            LatitudePred = predicting_latitude_on_validationset,
                            FloorError = validation_data$FLOOR - predicting_floor_on_validationset,
                            LongitudeError = validation_data$LONGITUDE - predicting_longitude_on_validationset,
                            LatitudeError = validation_data$LATITUDE - predicting_latitude_on_validationset)

hist(errorDataFrame$FloorError)
hist(errorDataFrame$LongitudeError)
hist(errorDataFrame$LatitudeError)
# Errors seem to be evenly distributed
plot(validation_data$LONGITUDE, abs(errorDataFrame$LongitudeError))
plot(errorDataFrame$LatitudeError)
# There seems to be some outliers in longitude and latitude, as most errors lie between -20 and 20.
# 

errorDataFrame %>% group_by(Floor) %>% summarise(median = abs(median(LatitudeError, na.rm = T)), mean = abs(mean(LatitudeError)))
mean(errorDataFrame$LatitudeError)
names(errorDataFrame)
class(errorDataFrame$LatitudeError)

# Model with cascading ####

# Start by predicting building

training_set$BUILDINGID<- as.factor(training_set$BUILDINGID)
testing_set$BUILDINGID<- as.factor(testing_set$BUILDINGID)

#randomForestBuilding<- randomForest(y = training_set$BUILDINGID, x = train_mod, ntree = 150)
#saveRDS(randomForestBuilding, file = "Model_random_forest_building")
buidingOnTest<- predict(randomForestBuilding, newdata = testing_set[, 1:465])
buidingOnTest
confusionMatrix(buidingOnTest, testing_set$BUILDINGID)
# Accuracy = 0.9982 Kappa = 0.9971
validation_data$BUILDINGID<- as.factor(validation_data$BUILDINGID)

buildingOnValidation<- predict(randomForestBuilding, newdata = validation_data[, c(1:520, 524)])
confusionMatrix(validation_data$BUILDINGID, buildingOnValidation)
# Accuracy = 0.9982 Kappa = 0.9972

train_mod$BUILDINGID<- randomForestBuilding$predicted


#cascadingRandomForestLongitude<- randomForest(y = training_set$LONGITUDE, x = train_mod, ntree = 150)
#saveRDS(cascadingRandomForestLongitude, file = "Model_cascading_random_forest_longitude")
longitudeOnTest<- predict(cascadingRandomForestLongitude, newdata = testing_set[, c(1:465, 469)])
longitudeOnTest
postResample(testing_set$LONGITUDE, longitudeOnTest)

longitudeOnValidation<- predict(cascadingRandomForestLongitude, newdata = validation_data[, c(1:521, 524)])
postResample(validation_data$LONGITUDE, longitudeOnValidation)
# RMSE = 11.38 R2 = 0.9911 MAE = 7.871

train_mod$LONGITUDE<- cascadingRandomForestLongitude$predicted

#cascadingRandonForestLatitude<- randomForest(y = training_set$LATITUDE, x = train_mod, ntree = 150)
#saveRDS(cascadingRandonForestLatitude, file = "Model_cascadingRandomForestLatitude")
latitudeOnTest<- predict(cascadingRandonForestLatitude, newdata = testing_set[, c(1:466, 469)])
latitudeOnTest
postResample(testing_set$LATITUDE, latitudeOnTest)

latitudeOnValidation<- predict(cascadingRandonForestLatitude, newdata = validation_data[, c(1:522, 524)])
postResample(validation_data$LATITUDE, latitudeOnValidation)
# RMSE = 9.733 R2 = 0.9823 MAE = 6.41

train_mod$LATITUDE<- cascadingRandonForestLatitude$predicted

#cascadingRandomForestFloor<- randomForest(y = training_set$FLOOR, x = train_mod, ntree = 150)
#saveRDS(cascadingRandomForestFloor, file = "Model_cascadingRandomForestFloor")
validation_data$FLOOR<- as.factor(validation_data$FLOOR)

floorOnTest<- predict(cascadingRandomForestFloor, newdata = testing_set[, c(1:467, 469)])
floorOnTest
confusionMatrix(testing_set$FLOOR, floorOnTest)

floorOnValidation<- predict(cascadingRandomForestFloor, newdata = validation_data[, c(1:524)])
confusionMatrix(validation_data$FLOOR, floorOnValidation)

# Modeling with KNN ####

# Cleaning the dataset for RSSI values that are too good or too weak
clean<- replace.value(data = training_set, names = 1:465, from = 100, to = -500)
clean$max<- apply(clean[, 1:465], MARGIN = 1, max)

clean<- clean %>% group_by(max) %>% filter(max< -30)

clean<- replace.value(data = clean, names = 1:465, from = -500, to = 100)
clean<- clean %>% group_by(max) %>% filter(max >= -85)
clean$max<- NULL


trainingKNN<- clean
trainingKNN<- trainingKNN[, -470]

cols<- names(validation_data) %in% names(trainingKNN)
validation<- validation_data[, cols]


validation<- replace.value(data = validation, names = 1:465, from = 100, to = -500)
validation$max<- apply(validation[, 1:465], MARGIN = 1, max)
validation<- validation %>% group_by(max) %>% filter(max < -30)
validation<- replace.value(data = validation, names = 1:465, from = -500, to = 100)
validation<- validation %>% group_by(max) %>% filter(max >= -85)
validation$max<- NULL

testingKNN<- validation

# Predicting the building with KNN
buildingIDknn<- knn(train = trainingKNN[, 1:465], test = testingKNN[, 1:465], cl = trainingKNN$BUILDINGID,
                    k = 3, prob = TRUE, algorithm = c("kd_tree", "cover_tree", "brute"))

buildingIDknn
confusionMatrix(buildingIDknn, validation$BUILDINGID)
# Acc = 0.9928, Kappa = 0.9886

# Building a KNN regression model for predicting latitude



#latitudeKNN<- knn.reg(train = trainingKNN[, 1:465], test = testingKNN[, 1:465],
#  y = trainingKNN$LATITUDE, k = 3, algorithm = c("kd_tree", "cover_tree",
#        "brute"))
latitudeKNN
saveRDS(latitudeKNN, file = "LatitudeKNN")
postResample(latitudeKNN$pred, validation$LATITUDE)
# RMSE = 12.19 R2 = 0.9707 MAE = 7.85
# Building a KNN for prediction Longitude


#longitudeKNN<- knn.reg(train = trainingKNN[, c(1:465)], test = testingKNN[, c(1:465)],
#   y = trainingKNN$LONGITUDE, k = 3, algorithm = c("kd_tree", "cover_tree",
#     "brute"))

longitudeKNN
saveRDS(longitudeKNN, file = "LongitudeKNN")
postResample(longitudeKNN$pred, validation$LONGITUDE)
# RMSE = 12.08 R2 = 0.9900 MAE = 8.16

# Building a KNN for predicting floor


#floorKNN<- knn(train = trainingKNN[, c(1:465)], test = testingKNN[, c(1:465)],
#  cl = trainingKNN$FLOOR, k = 3, prob = TRUE, algorithm = c("kd_tree", "cover_tree",
#      "brute"))
floorKNN
saveRDS(floorKNN, file = "FloorKNN")
confusionMatrix(floorKNN, validation$FLOOR)
# Acc = 0.7552 Kappa = 0.6687














