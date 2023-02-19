# Load libraries, register cores
library(data.table)
library(ranger)
library(arf)
library(Rfast)
library(ggplot2)
library(ggsci)
library(cowplot)
library(reticulate)
library(doMC)
registerDoMC(20)

# Load GeFs
np <- reticulate::import("numpy")
source_python('gefs_fun.py')

# Set seed
set.seed(123, "L'Ecuyer-CMRG")

# Simulation script

sim_exp <- function(b, n, d, sparsity) {
  # Effects 
  beta <- double(length = d)
  k <- round((1 - sparsity) * d)
  if (k > 0) {
    beta[1:k] <- 1
  }
  
  # Parameters
  mu <- rep(0, d)
  sigma <- toeplitz(0.9^(0:(d-1)))
  
  # Create data
  x <- matrix(rmvnorm(n = n + 1000, mu = mu, sigma = sigma), ncol = d,
              dimnames = list(NULL, paste0('x', 1:d)))
  y <- rbinom(n + 1000, size = 1, prob = plogis(x %*% beta))
  
  # Split train/test
  trn_x <- x[1:n, ]
  trn_y <- y[1:n]
  tst_x <- x[(n + 1):(n + 1000), ]
  tst_y <- y[(n + 1):(n + 1000)]
  
  # Adversarial RF
  arf <- adversarial_rf(trn_x, num_trees = 100, verbose = FALSE, parallel = FALSE)
  # Truncated normal density
  fd_tnorm <- forde(arf, x_trn = trn_x, x_tst = tst_x, family = 'truncnorm',
                    parallel = FALSE)
  # PWC unsupervised
  fd_pwc_u <- forde(arf, x_trn = trn_x, x_tst = tst_x, family = 'unif',
                    parallel = FALSE)

  # Competition
  trn_dat <- data.table(y = trn_y, trn_x)
  rf <- ranger(y ~ ., data = trn_dat, num.trees = 100, min.node.size = 5,
               keep.inbag = TRUE, classification = TRUE, num.threads = 1)
  # PWC supervised
  fd_pwc_s <- forde(rf, x_trn = trn_x, x_tst = tst_x, family = 'unif',
                    prune = FALSE, parallel = FALSE)
  
  # Correia
  ncat <- reticulate::np_array(c(rep(1L, d), 2L), dtype = np$int64)
  trn_y_py <- reticulate::np_array(trn_y, dtype = np$double)
  tst_dat <- cbind(tst_x, tst_y)
  loglik_gefs <- gefs_fun(trn_x, trn_y_py, tst_dat, ncat, 100L)
  
  # Results
  out <- data.table(
    'b' = b, 'n' = n, 'd' = d, 'sparsity' = sparsity,
    'ARF' = -mean(fd_tnorm$loglik), 'PWCu' = -mean(fd_pwc_u$loglik),
    'GeF' = -mean(loglik_gefs[loglik_gefs < 0]), 'PWCs' = -mean(fd_pwc_s$loglik)
  )
  return(out)
}

# By sample size
df1 <- foreach(bb = 1:20, .combine = rbind) %:%
  foreach(nn = round(10^(seq(2, 4, length.out = 10))), .combine = rbind) %dopar%
  sim_exp(b = bb, n = nn, d = 10, sparsity = 1/2)

# By signal sparsity
df2 <- foreach(bb = 1:20, .combine = rbind) %:%
  foreach(sp = seq(0, 1, by = 0.1), .combine = rbind) %dopar%
  sim_exp(b = bb, n = 2000, d = 10, sparsity = sp)


df <- rbind(df1, df2)
saveRDS(df, 'NLL_exp.rds')

# Plot: NLL by sample size
tmp0 <- melt(df1, id.vars = c('b', 'n'), 
             measure.vars = c('ARF', 'PWCu', 'GeF', 'PWCs'),
             variable.name = 'Method', value.name = 'NLL')
tmp <- tmp0[, mean(NLL), by = .(n, Method)]
colnames(tmp)[3] <- 'NLL'
tmp[, se := tmp0[, sd(NLL), by = .(n, Method)]$V1]
tmp[, Method := factor(Method, 
                       levels = c("PWCs", "PWCu", "GeF", "ARF"), 
                       labels = c("PWC\n(sup.)", "PWC\n(unsup.)", "GeFs", "FORGE"))]
p1 <- ggplot(tmp, aes(n, NLL, shape = Method, color = Method, fill = Method, 
                ymin = NLL - se, ymax = NLL + se)) + 
  geom_point() +
  geom_path() + 
  geom_ribbon(alpha = 0.1, color = NA) + 
  scale_x_log10() + 
  scale_y_continuous(breaks = pretty_breaks()) + 
  scale_shape_manual(values = c(16, 3, 17, 15)) + 
  scale_color_nejm() + 
  scale_fill_nejm() + 
  xlab("Sample size") + 
  theme_bw()

# Plot: NLL by sparsity
tmp0 <- melt(df2, id.vars = c('b', 'sparsity'), 
             measure.vars = c('ARF', 'PWCu', 'GeF', 'PWCs'),
             variable.name = 'Method', value.name = 'NLL')
tmp <- tmp0[, mean(NLL), by = .(sparsity, Method)]
colnames(tmp)[3] <- 'NLL'
tmp[, se := tmp0[, sd(NLL), by = .(sparsity, Method)]$V1]
tmp[, Method := factor(Method, 
                       levels = c("PWCs", "PWCu", "GeF", "ARF"), 
                       labels = c("PWC\n(sup.)", "PWC\n(unsup.)", "GeFs", "FORGE"))]
p2 <- ggplot(tmp, aes(sparsity, NLL, shape = Method, color = Method, fill = Method, 
                ymin = NLL - se, ymax = NLL + se)) + 
  geom_point() +
  geom_path() + 
  geom_ribbon(alpha = 0.1, color = NA) + 
  scale_y_continuous(breaks = pretty_breaks()) + 
  scale_shape_manual(values = c(16, 3, 17, 15)) + 
  scale_color_nejm() + 
  scale_fill_nejm() + 
  xlab("Sparsity") + 
  theme_bw() + 
  theme(axis.title.y = element_blank(), 
        legend.text = element_text(lineheight = .8), 
        legend.key.height=unit(22, "pt"))

# Plot together -----------------------------------------------------------
prow <- plot_grid(p1 + theme(legend.position = "none"), 
                  p2 + theme(legend.position = "none"), 
                  ncol = 2, labels = "AUTO", label_x = c(.05, 0), 
                  rel_widths = c(.513, .487))
legend <- get_legend(
  p1 + theme(legend.position = "bottom")
)
cowplot::plot_grid(prow, legend, ncol = 1, rel_heights = c(1, .1))
ggsave("NLL_exp.pdf", width = 7, height = 3)




