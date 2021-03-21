#as.vector(installed.packages()[,"Package"])

#which(grepl(pattern = "^h2o*", x = as.vector(installed.packages()[,"Package"]), ignore.case = TRUE))

library(tidyverse)

data = read_rds(file = "./df2_all_ft.rds")

head(data)

library(h2o)
localH2O = h2o.init()
demo(h2o.kmeans)

data %>% 
count(Continuation_rates) %>% 
mutate(prop = n/sum(n))

set.seed(123)
train_idx <- caret::createDataPartition(y = data$Continuation_rates, p = 0.8, list = FALSE)

train_data <- data[train_idx,]
test_data <- data[-train_idx,]

train_x <- train_data %>% select(-Continuation_rates)
train_y <- train_data %>% select(Continuation_rates)

test_x <- test_data %>% select(-Continuation_rates)
test_y <- test_data %>% select(Continuation_rates)

outcome <- "Continuation_rates"

predictors <- setdiff(colnames(data),outcome)

#Convert to h2oFrame object
train_data <- as.h2o(train_data)
test_data <- as.h2o(test_data)

doParallel::registerDoParallel(parallel::detectCores())

system.time(
h2o_model <- h2o.automl(x = predictors, 
                        y = outcome,
                        training_frame = train_data,
                        validation_frame = test_data, 
                        nfolds = 10, 
                        balance_classes = TRUE, 
                        max_models = 20, 
                        seed = 123, 
                        max_runtime_secs = 60*30)
)

doParallel::stopImplicitCluster()

training_results <- as_tibble(h2o_model@leaderboard) %>% mutate(sn = row_number()) %>% 
relocate(sn, `.before` = everything())

bind_rows(
training_results %>% head(1),

training_results %>% 
filter(str_detect(string = model_id, pattern = "Ens")))

h2o_model@leader

#ROC OPERATING POINT
class(h2o_model@leader)

perf <- h2o.performance(model = h2o_model@leader, newdata = test_data, valid = TRUE)

plot(perf, type="roc")

perf

metrics <- as_tibble(bind_cols(h2o.tpr(object = perf),h2o.fpr(object = perf)))

distances <- metrics %>% 
select(threshold...1,tpr,fpr) %>% 
rename(threshold = threshold...1,
       sensitivity = tpr) %>%  
mutate(specificity = 1-fpr) %>% 
mutate(d = sqrt(((1-sensitivity)^2)+((1-specificity)^2))) %>% 
arrange(d)

head(distances)

v <- distances$fpr[1]
h <- distances$sensitivity[1]
thr <- distances$threshold[1]

plot(perf, type="roc")
abline(v = v, col = "red")
abline(h = h, col = "red")

predictions <- predict(object = h2o_model@leader, newdata = test_data, type = "prob")

adjusted_preds <- as_tibble(predictions) %>% 
bind_cols(test_y) %>% 
mutate(new_predict = ifelse(test = High_Continuation<thr, yes = "Low_Continuation", no = "High_Continuation"),
       new_predict = as.factor(new_predict))

adjusted_preds %>% 
summarise(accuracy = mean(Continuation_rates==new_predict))

caret::confusionMatrix(reference = adjusted_preds$Continuation_rates, 
                       data = adjusted_preds$new_predict)

#Accuracy for the low continuation category (specificity)
64/(64+17)

#High continuation accuracy (sensitivity)
18/(18+19)

h2o.auc(object = perf)

#PRC OPERATING POINT
plot(perf, type="pr")

ls()

pr_distances <- bind_cols(as_tibble(h2o.precision(object = perf)),
as_tibble(h2o.recall(object = perf))) %>% 
select(thr = threshold...1, precision, tpr) %>% 
mutate(d = sqrt((1-tpr)^2+(1-precision)^2)) %>% 
arrange(d)

head(pr_distances)

y <- pr_distances$precision[1]
x <- pr_distances$tpr[1]
pr_thr <- pr_distances$thr[1]

plot(perf, type="pr")
abline(v = x, col="red")
abline(h = y, col="red")

pr_predictions <- predict(object = h2o_model@leader, newdata = test_data, type = "probs")

head(pr_predictions)

pr_adj_preds <- pr_predictions %>% 
as_tibble() %>% 
mutate(adj_pred = ifelse(test = Low_Continuation<pr_thr, yes = "High_Continuation", no = "Low_Continuation"),
       adj_pred = as.factor(adj_pred)) %>% 
bind_cols(test_y)

pr_adj_preds %>% 
summarise(accuracy = mean(adj_pred==Continuation_rates))

caret::confusionMatrix(data = pr_adj_preds$adj_pred, reference = pr_adj_preds$Continuation_rates)

#High continuation accuracy (sensitivity)
10/(10+27)

#Low continuation accuracy (specificity)
77/(77+4)

h2o.aucpr(object = perf)

save.image("./h2o_ml.RData")
