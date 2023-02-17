library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(xtable)
files <- c(list.files(path = "./benchmark_results",
                      pattern = ".*\\.csv$",
                      all.files = FALSE, full.names = TRUE, recursive = FALSE, ignore.case = TRUE,
                      include.dirs = FALSE, no.. = FALSE))
df <- data.frame()
for (i in 1:length(files)) {
  df <- bind_rows(df, read.csv(files[i]))
} 

helper_fun <- function(f1_char, first = T){
  list_f1 = gsub('[', "",f1_char , fixed = T) %>% gsub(']', "", ., fixed = T) %>% 
    strsplit(., split = " ") %>% lapply(as.numeric) %>% lapply(., function(x) {na.omit(x)[1:2]})
  if(first){return(unlist(lapply(list_f1, function(x) x[1])))}
  else{return(unlist(lapply(list_f1, function(x) x[2])))}
}

df = df %>% mutate(f1_1 = helper_fun(f1_none, first = T)) %>% 
  mutate(f1_2 = helper_fun(f1_none, first = F)) %>% select(-c(f1_none, X, process_time)) # separate f1 scores: f1_1, f1_2

# for adult ->  higher f1 score corresponds to positive class '>50k'
# for census -> f1_2 corresponds to positive class '+50000'
# for credit -> f1_2 corresponds to positive class '1'

df$f1_pos = df$f1_2
df[df$dataset == "adult", "f1_pos" ]=apply(df[df$dataset == "adult",c("f1_1", "f1_2")], 1, function(x){max(x)}) # detect max f1

df = df %>% group_by(wall_time) %>% # calculate mean values for each replication
  mutate(f1_pos_mean = mean(f1_pos),
         f1_2_mean = mean(f1_2),
         accuracy_score_mean = mean(accuracy_score),
         f1_macro_mean = mean(f1_macro),
         f1_micro_mean = mean(f1_micro)) %>%
  select(-c(classifier, accuracy_score, f1_1, f1_2,f1_pos,f1_micro, f1_macro)) %>% # drop scores for each classifier
  unique()

df = df %>% group_by(dataset, model) %>% # calculate mean and sd values across replications
  mutate(f1_pos_sd = sd(f1_pos_mean), 
         f1_pos_mean = mean(f1_pos_mean),
         f1_2_sd = sd(f1_2_mean), 
         f1_2_mean = mean(f1_2_mean),
         accuracy_score_sd = sd(accuracy_score_mean), 
         accuracy_score_mean = mean(accuracy_score_mean),
         f1_macro_sd = sd(f1_macro_mean), f1_macro_mean = mean(f1_macro_mean),
         f1_macro_sd = sd(f1_micro_mean), f1_micro_mean = mean(f1_micro_mean),
         mean_wall_time = mean(wall_time)) %>%
  select(-wall_time) %>%
  unique() # drop scores for each replication

# full table
xtable_df = df %>% mutate(id = ifelse(model =="oracle", 1, ifelse(model=="gen_rf", 2, ifelse(model=="CTGAN", 3, 4)))) %>% 
  group_by(dataset) %>% arrange(dataset,id) %>% relocate(f1_pos_mean, .after = accuracy_score_mean) %>% 
  select(-id)

# table to latex
print(xtable(xtable_df), include.rownames = F)

# merge F1 scores -- always use lowest score for F1 in binary classification, F1 micro for multiclass classification
xtable_f1 = df %>% mutate(id = ifelse(model =="oracle", 1, ifelse(model=="gen_rf", 2, ifelse(model=="CTGAN", 3, 4))))%>%
  group_by(dataset) %>% arrange(dataset,id) 
xtable_f1$F1 = c()
xtable_f1[xtable_f1$dataset == "adult","F1"] = xtable_f1[xtable_f1$dataset == "adult", "f1_pos_mean"]
xtable_f1[xtable_f1$dataset == "census","F1"] = xtable_f1[xtable_f1$dataset == "census", "f1_2_mean"]
xtable_f1[xtable_f1$dataset == "covtype","F1"] = xtable_f1[xtable_f1$dataset == "covtype", "f1_macro_mean"]
xtable_f1[xtable_f1$dataset == "credit","F1"] = xtable_f1[xtable_f1$dataset == "credit", "f1_2_mean"]
xtable_f1[xtable_f1$dataset == "intrusion","F1"] = xtable_f1[xtable_f1$dataset == "intrusion", "f1_macro_mean"]
xtable_f1[xtable_f1$dataset == "mnist12","F1"] = xtable_f1[xtable_f1$dataset == "mnist12", "f1_macro_mean"]
xtable_f1[xtable_f1$dataset == "mnist28","F1"] = xtable_f1[xtable_f1$dataset == "mnist28", "f1_macro_mean"]

xtable_f1$F1_sd = c()
xtable_f1[xtable_f1$dataset == "adult","F1_sd"] = xtable_f1[xtable_f1$dataset == "adult", "f1_pos_sd"]
xtable_f1[xtable_f1$dataset == "census","F1_sd"] = xtable_f1[xtable_f1$dataset == "census", "f1_2_sd"]
xtable_f1[xtable_f1$dataset == "covtype","F1_sd"] = xtable_f1[xtable_f1$dataset == "covtype", "f1_macro_sd"]
xtable_f1[xtable_f1$dataset == "credit","F1_sd"] = xtable_f1[xtable_f1$dataset == "credit", "f1_2_sd"]
xtable_f1[xtable_f1$dataset == "intrusion","F1_sd"] = xtable_f1[xtable_f1$dataset == "intrusion", "f1_macro_sd"]
xtable_f1[xtable_f1$dataset == "mnist12","F1_sd"] = xtable_f1[xtable_f1$dataset == "mnist12", "f1_macro_sd"]
xtable_f1[xtable_f1$dataset == "mnist28","F1_sd"] = xtable_f1[xtable_f1$dataset == "mnist28", "f1_macro_sd"]

xtable_f1$F1_sd = paste0("(", format(round(xtable_f1$F1_sd, digits = 3),nsmall = 3), ")")
xtable_f1$accuracy_score_sd = paste0("(", format(round(xtable_f1$accuracy_score_sd, digits = 3),nsmall = 3), ")")

xtable_f1$characteristics = c()
xtable_f1[xtable_f1$dataset == "adult" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 32,561"
xtable_f1[xtable_f1$dataset == "adult" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 14"
xtable_f1[xtable_f1$dataset == "adult" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 2"
xtable_f1[xtable_f1$dataset == "census" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 298,006"
xtable_f1[xtable_f1$dataset == "census" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 40"
xtable_f1[xtable_f1$dataset == "census" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 2"
xtable_f1[xtable_f1$dataset == "covtype" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 581,012"
xtable_f1[xtable_f1$dataset == "covtype" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 54"
xtable_f1[xtable_f1$dataset == "covtype" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 7"
xtable_f1[xtable_f1$dataset == "credit" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 284,807"
xtable_f1[xtable_f1$dataset == "credit" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 30"
xtable_f1[xtable_f1$dataset == "credit" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 2"
xtable_f1[xtable_f1$dataset == "intrusion" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 494,021"
xtable_f1[xtable_f1$dataset == "intrusion" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 40"
xtable_f1[xtable_f1$dataset == "intrusion" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 5"
xtable_f1[xtable_f1$dataset == "mnist12" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 70,000"
xtable_f1[xtable_f1$dataset == "mnist12" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 144"
xtable_f1[xtable_f1$dataset == "mnist12" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 10"
xtable_f1[xtable_f1$dataset == "mnist28" & xtable_f1$model == "gen_rf" ,"characteristics"] = "$n$ = 70,000"
xtable_f1[xtable_f1$dataset == "mnist28" & xtable_f1$model == "CTGAN" ,"characteristics"] = "$p$ = 784"
xtable_f1[xtable_f1$dataset == "mnist28" & xtable_f1$model == "oracle" ,"characteristics"] = "classes = 10"

xtable_f1$model[xtable_f1$model == "gen_rf"] = "FORGE"
xtable_f1 = xtable_f1 %>%relocate(characteristics, .after = dataset) %>% relocate(F1, .after = accuracy_score_mean) %>% 
  select(c(dataset, characteristics, model, accuracy_score_mean,accuracy_score_sd, F1,F1_sd, mean_wall_time)) %>% 
 rename(accuracy = accuracy_score_mean) %>% rename(time = mean_wall_time)
# table to latex
print(xtable(xtable_f1, digits = 3), include.rownames = F,  booktabs = T)

