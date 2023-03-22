# Load libraries, register cores
library(data.table)
library(arf)
library(doMC)
registerDoMC(10)

# Set seed
set.seed(123, "L'Ecuyer-CMRG")

# Output file
df <- data.table(
  'dataset' = NA_character_, 'iter' = NA_integer_,
  'n' = NA_real_, 'd' = NA_real_, 'NLL' = NA_real_
)
saveRDS(df, './experiments/NLL_binary.rds')

# Likelihood evaluation function
nll_fn <- function(dataset, iter) {
  # Load data
  trn <- fread(paste0('./data/', dataset, '/', dataset, '.train.data'))
  val <- fread(paste0('./data/', dataset, '/', dataset, '.valid.data'))
  trn <- rbind(trn, val)
  tst <- fread(paste0('./data/', dataset, '/', dataset, '.test.data'))
  n <- nrow(trn)
  d <- ncol(trn)
  colnames(trn) <- colnames(tst) <- paste0('x', 1:d)
  # Train
  if (dataset == 'c20ng') {
    b <- 100
  } else {
    b <- 200
  }
  cat(paste0('Dataset: ', dataset, '\n'))
  suppressWarnings(
    arf <- adversarial_rf(trn, num_trees = b, min_node_size = 4)
  )
  suppressWarnings(
    psi <- forde(arf, trn, alpha = 0.05)
  )
  ll <- lik(arf, psi, tst, batch = 5000)
  # Export
  out <- data.table(
    'dataset' = dataset, 'iter' = iter, 'n' = n, 'd' = d, 'NLL' = -mean(ll)
  )
  df <- readRDS('./experiments/NLL_binary.rds')
  df <- rbind(df, out)
  saveRDS(df, './experiments/NLL_binary.rds')
}

# Loop through data, compute negative log-likelihood
datasets <- list.files('./data')
datasets <- datasets[!grepl('amzn', datasets)]
foreach(d = datasets) %:%
  foreach(i = 1:5) %do% nll_fn(d, i)

# Polish
df <- readRDS('./experiments/NLL_binary.rds')
names <- c('nltcs', 'msnbc', 'kdd', 'plants', 'audio', 'jester', 'netflix', 
           'accidents', 'retail', 'pumsb', 'dna', 'kosarek', 'msweb', 'book', 
           'movie', 'webkb', 'reuters', '20ng', 'bbc', 'ad')
df[dataset == 'baudio', dataset := 'audio']
df[dataset == 'bnetflix', dataset := 'netflix']
df[dataset == 'cwebkb', dataset := 'webkb']
df[dataset == 'pumsb_star', dataset := 'pumsb']
df[dataset == 'tmovie', dataset := 'movie']
df[dataset == 'tretail', dataset := 'retail']
df[dataset == 'cr52', dataset := 'reuters']
df[dataset == 'c20ng', dataset := '20ng']
df <- na.omit(df)

# Compute mean and standard error
df[, mean := mean(NLL), by = dataset]
df[, se := sd(NLL), by = dataset] 
df <- unique(df[, .(dataset, mean, se)])
colnames(df)[2] <- 'NLL'
df <- df[order(match(dataset, names))]


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

