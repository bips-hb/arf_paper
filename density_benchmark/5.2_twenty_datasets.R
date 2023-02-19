# Load libraries, register cores
library(data.table)
library(arf)
library(doMC)
registerDoMC(10)

# Set seed
set.seed(123, "L'Ecuyer-CMRG")

# Output file
df <- data.table('dataset' = NA_character_, 'n' = NA_real_, 'd' = NA_real_, 
                 'NLL' = NA_real_)
saveRDS(df, './experiments/NLL_binary.rds')

# Likelihood evaluation function
nll_fn <- function(dataset) {
  # Load data
  trn <- fread(paste0('./data/', dataset, '/', dataset, '.train.data'))
  val <- fread(paste0('./data/', dataset, '/', dataset, '.valid.data'))
  tst <- fread(paste0('./data/', dataset, '/', dataset, '.test.data'))
  n <- nrow(trn)
  d <- ncol(trn)
  colnames(trn) <- colnames(val) <- colnames(tst) <- paste0('x', 1:d)
  # Train
  cat(paste0('Dataset: ', dataset, '\n'))
  suppressWarnings(
    arf <- adversarial_rf(trn, num_trees = 100)
  )
  suppressWarnings(
    psi <- forde(arf, val)
  )
  ll <- lik(arf, psi, tst, batch = 5000)
  # Export
  out <- data.table('dataset' = dataset, 'n' = n, 'd' = d, 'NLL' = -mean(ll))
  df <- readRDS('./experiments/NLL_binary.rds')
  df <- rbind(df, out)
  saveRDS(df, './experiments/NLL_binary.rds')
}

# Loop through data, compute negative log-likelihood
datasets <- list.files('./data')
datasets <- datasets[!grepl('amzn', datasets)]
foreach(d = datasets) %do% nll_fn(d)


# Meta data
dat_summary <- function(dataset) {
  trn <- fread(paste0('./data/', dataset, '/', dataset, '.train.data'))
  val <- fread(paste0('./data/', dataset, '/', dataset, '.valid.data'))
  tst <- fread(paste0('./data/', dataset, '/', dataset, '.test.data'))
  data.table('Dataset' = dataset, 'Train' = nrow(trn), 'Validation' = nrow(val),
             'Test' = nrow(tst), 'Dimensionality' = ncol(trn))
}
df <- foreach(d = datasets, .combine = rbind) %do% dat_summary(d)
df <- df[!grepl('amzn', datasets)]
df <- df[order(Train)]

