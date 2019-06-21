library(jsonlite)
library(brms)


standardize <- function(c, scale=F) {
    if (scale) {
        (c - mean(c)) / sd(c)
    } else {
        c - mean(c)
    }
}

parameters <- c(
    "example_type", "cell", "emb_dim", "hid_dim", "bidirectional",
    "n_layers", "dropout", "cutoff_cosine", "margin",
    "weight", "lr", "score", "rnn_dropout", "forget_bias"
)

read_log <- function(fpath) {
    df <- stream_in(file(fpath), verbose=F)
    score <- df$dev_score
    df <- df$params
    df$score <- score
    df[,parameters]
}

red <- "#e74c3c"
blue <- "#3498db"

df <- read_log("../log/FSINST-rnn.jsonl")
df$dropout_c <- standardize(df$dropout)
df$hid_dim_c <- standardize(log(df$hid_dim))
df$margin_c <- standardize(df$margin)
df$bidirectional <- factor(df$bidirectional, labels=c("no", "yes"))
df$cutoff_cosine <- factor(df$cutoff_cosine, labels=c("no", "yes"))
df$example_type <- as.factor(df$example_type)
df$cell <- as.factor(df$cell)


m <- brm(score ~ example_type * margin_c + dropout_c + bidirectional + cell + hid_dim_c,
         data=df, family="gaussian",
         prior=c(set_prior("normal(0, 1)", class="b"),
                 set_prior("normal(0, 1)", class="Intercept"),
                 set_prior("cauchy(0, 1)", class="sigma")),
         sample_prior = TRUE)
summary(m)

p_margin <- plot(marginal_effects(m, effects=c("margin_c:example_type")))[[1]]
p_margin <- p_margin + xlab("margin (centered)") + ylab("Mean Average Precision")
p_margin <- p_margin + theme_light(base_size=15) +
    theme(legend.position = "top",
          plot.margin=margin(0, 0.5, 0, 0, "cm"),
          legend.margin=margin(t = 0, unit='cm')) + 
    scale_colour_manual("Loss", values=c(red, blue), labels=c("duplet", "triplet")) +
    scale_fill_manual("Loss", values=c(red, blue), labels=c("duplet", "triplet"))
ggsave("../analyses/margin_example_type.pdf", p_margin, width=6, height=4)

conditions <- data.frame(cell=c("LSTM", "GRU"))
p_gru <- plot(marginal_effects(m, effects=c("example_type:bidirectional"),
                               conditions=conditions))[[1]]
p_gru$data$cond__ <- factor(p_gru$data$cond__, labels=c("LSTM", "GRU"))
p_gru <- p_gru + xlab("Loss") + ylab("Mean Average Precision") +
    scale_x_discrete(labels=c("duplet", "triplet")) +
    scale_colour_manual("Bidirectional", values=c(red, blue), labels=c("no", "yes")) +
    scale_fill_manual("Bidirectional", values=c(red, blue), labels=c("no", "yes")) + 
    theme_light(base_size=15) +
    theme(legend.position = "top",
          plot.margin=margin(0, 0, 0, 0, "cm"),
          legend.margin=margin(t = 0, unit='cm'))
ggsave("../analyses/rnn-bidirectional.pdf", p_gru, width=6, height=4)


library(bayesplot)
color_scheme_set("viridis")
post <- posterior_samples(m, add_chain = T)
colnames(post) <- c('gamma', 'beta[l]', 'beta[m]', 'beta[d]', 'beta[b]',
                    'beta[c]', 'beta[h]', 'beta[l][m]', 'sigma',
                    'p', 'ps', 'lp__', 'chain', 'iter')
M <- mcmc_trace(post[, c(1:9, 13)], 
           facet_args = list(ncol = 3, labeller=ggplot2::label_parsed), 
           size = .15) +
  labs(title = "Trace plots of the HMC") +
    theme_minimal() +
    theme(text = element_text(size=18),
          plot.margin = margin(1, 1, 1, 1, "cm"))
ggsave("../analyses/test-trace.pdf", M, width=12)

M <- mcmc_dens(post[, c(1:9, 13)], 
           facet_args = list(ncol = 3, labeller=ggplot2::label_parsed), 
           size = .15) +
  labs(title = "Posterior Distribution Density Plots") +
    theme_minimal() +
    theme(text = element_text(size=18),
          plot.margin = margin(1, 1, 1, 1, "cm"))
ggsave('../analyses/dens-plots.pdf', M, width=12)
