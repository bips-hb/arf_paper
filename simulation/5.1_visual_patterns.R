# Load libraries, register cores, set seed
library(data.table)
library(arf)
library(fdm2id)
library(mlbench)
library(ggplot2)
library(ggsci)
library(doMC)
registerDoMC(8)
set.seed(1, "L'Ecuyer-CMRG")

# Simulation function
sim_fun <- function(n_trn, n_tst, dataset) {
  # Simulate data
  if (dataset == 'twomoons') {
    x <- data.twomoons(n = n_trn/2, graph = FALSE)
    x$Class <- gsub('Class ', '', x$Class)
  } else {
    if (dataset == 'cassini') {
      tmp <- mlbench.cassini(n_trn)
    } else if (dataset == 'smiley') {
      tmp <- mlbench.smiley(n_trn)
    } else if (dataset == 'shapes') {
      tmp <- mlbench.shapes(n_trn)
    }
    x <- data.frame(tmp$x, tmp$classes)
    colnames(x) <- c('X', 'Y', 'Class')
  }
  # Fit model, generate data
  arf <- adversarial_rf(x, num_trees = 20, mtry = 2, verbose = FALSE)
  psi <- forde(arf, x)
  synth <- forge(psi, n_tst)
  # Put it all together, export
  df_orig <- data.table(Data = "Original", x[sample(n_trn, n_tst), ])
  rbind(df_orig, data.table(Data = "Synthetic", synth))[, Dataset := dataset]
}

# Execute in parallel
dsets <- c('twomoons', 'cassini', 'smiley', 'shapes')
df <- foreach(d = dsets, .combine = rbind) %dopar% sim_fun(2000, 1000, d)

# Set scales free but fix x-axis ticks

# Scatter plot
ggplot(df, aes(x = X, y = Y, color = Class, shape = Class)) + 
  geom_point(alpha = 0.75) + 
  scale_color_npg() + 
  facet_grid(Data ~ Dataset) + 
  theme_bw() + 
  theme(text = element_text(size = 14), legend.position = 'none')
ggsave(paste0("examples", ".pdf"), width = 8, height = 4)


