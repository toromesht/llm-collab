#!/usr/bin/env Rscript
# ─── Bayesian Router Core (R) ───────────────────────────────
# R handles the statistical learning. Python calls via:
#   Rscript R/bayesian_router.R --command learn --model ds-think --reward 1.0
#   Rscript R/bayesian_router.R --command route --question "Solve 2x+5=17"
#   Rscript R/bayesian_router.R --command plot --output pathways.png
#
# Advantages of R for this task:
#   • Built-in Beta distribution (dbeta, pbeta, rbeta) — no NumPy needed
#   • Built-in conjugate Bayesian updating
#   • ggplot2 for publication-quality visualization
#   • lm/glm for statistical modeling of model performance
#   • Native data.frame for tracking routing history
# ──────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

# ═══════════════════════════════════════════════════════════════
# BAYESIAN BETA-BERNOULLI ROUTER
# ═══════════════════════════════════════════════════════════════

MODELS <- c("ds-pro", "ds-think", "glm", "qwen", "kimi", "groq")
N_MODELS <- length(MODELS)

# Generative model: Beta(alpha, beta) posterior per model per context
# Stored as a named list for easy JSON serialization
posteriors <- list(
  math     = setNames(as.list(rep(1, N_MODELS)), MODELS),  # alpha
  math_b   = setNames(as.list(rep(1, N_MODELS)), MODELS),  # beta
  code     = setNames(as.list(rep(1, N_MODELS)), MODELS),
  code_b   = setNames(as.list(rep(1, N_MODELS)), MODELS),
  facts    = setNames(as.list(rep(1, N_MODELS)), MODELS),
  facts_b  = setNames(as.list(rep(1, N_MODELS)), MODELS),
  general  = setNames(as.list(rep(1, N_MODELS)), MODELS),
  general_b= setNames(as.list(rep(1, N_MODELS)), MODELS)
)

# Routing history
history <- data.frame(
  timestamp  = numeric(0),
  question   = character(0),
  context    = character(0),
  model      = character(0),
  reward     = numeric(0),
  stringsAsFactors = FALSE
)

# ─── Context detection ───────────────────────────────────────

detect_context <- function(question) {
  q <- tolower(question)
  if (any(sapply(c("solve", "prove", "theorem", "equation", "integral",
        "derivative", "matrix", "algebra", "calculus", "math"), grepl, q))) {
    return("math")
  }
  if (any(sapply(c("python", "code", "function", "sql", "algorithm",
        "program", "class", "implement"), grepl, q))) {
    return("code")
  }
  if (any(sapply(c("what", "who", "when", "where", "capital", "how many",
        "how much", "define", "explain"), grepl, q))) {
    return("facts")
  }
  return("general")
}

# ─── Conjugate Bayesian update ───────────────────────────────

bayesian_update <- function(context, model, reward) {
  alpha_key <- paste0(context, "")
  beta_key  <- paste0(context, "_b")

  # Alpha update: α' = α + r
  posteriors[[context]][[model]] <<-
    posteriors[[context]][[model]] + reward

  # Beta update: β' = β + (1 - r)
  posteriors[[paste0(context, "_b")]][[model]] <<-
    posteriors[[paste0(context, "_b")]][[model]] + (1 - reward)
}

# ─── Expected success probability ────────────────────────────

expected_success <- function(context, model) {
  a <- posteriors[[context]][[model]]
  b <- posteriors[[paste0(context, "_b")]][[model]]
  return(a / (a + b))
}

# ─── Thompson sampling ───────────────────────────────────────

thompson_sample <- function(context) {
  samples <- sapply(MODELS, function(m) {
    a <- posteriors[[context]][[m]]
    b <- posteriors[[paste0(context, "_b")]][[m]]
    return(rbeta(1, max(a, 0.1), max(b, 0.1)))
  })
  names(samples) <- MODELS
  return(samples)
}

# ─── UCB selection ───────────────────────────────────────────

ucb_select <- function(context, c = 2.0) {
  means <- sapply(MODELS, function(m) expected_success(context, m))
  counts <- sapply(MODELS, function(m) {
    posteriors[[context]][[m]] + posteriors[[paste0(context, "_b")]][[m]] - 2
  })
  total <- sum(counts)
  ucb <- means + c * sqrt(log(max(total, 1) + 1) / (counts + 1))
  return(MODELS[which.max(ucb)])
}

# ─── Route question ──────────────────────────────────────────

route <- function(question) {
  ctx <- detect_context(question)
  # Thompson sample 100 times, pick the most frequently winning model
  wins <- integer(N_MODELS); names(wins) <- MODELS
  for (i in 1:100) {
    ts <- thompson_sample(ctx)
    wins[names(which.max(ts))] <- wins[names(which.max(ts))] + 1
  }
  best <- MODELS[which.max(wins)]

  return(list(
    context        = ctx,
    primary_model  = best,
    expected_probs = sapply(MODELS, function(m) expected_success(ctx, m)),
    thompson_wins  = wins,
    ucb_model      = ucb_select(ctx)
  ))
}

# ─── Learn from outcome ──────────────────────────────────────

learn <- function(question, model, reward) {
  ctx <- detect_context(question)
  bayesian_update(ctx, model, reward)

  # Append to history
  history <<- rbind(history, data.frame(
    timestamp = Sys.time(),
    question  = substr(question, 1, 80),
    context   = ctx,
    model     = model,
    reward    = reward,
    stringsAsFactors = FALSE
  ))

  return(list(context = ctx, posteriors_updated = TRUE))
}

# ─── Summary statistics ──────────────────────────────────────

summarize <- function() {
  result <- list()
  for (ctx in c("math", "code", "facts", "general")) {
    result[[ctx]] <- data.frame(
      model = MODELS,
      alpha = unlist(posteriors[[ctx]][MODELS]),
      beta  = unlist(posteriors[[paste0(ctx, "_b")]][MODELS]),
      expected = sapply(MODELS, function(m) expected_success(ctx, m)),
      stringsAsFactors = FALSE
    )
  }
  result$total_episodes <- nrow(history)
  result$models_used <- as.list(table(history$model))
  result$avg_reward <- if(nrow(history) > 0) mean(history$reward) else 0
  return(result)
}

# ─── Plot: posterior distributions ───────────────────────────

plot_posteriors <- function(output_file = "posteriors.png") {
  df <- data.frame()
  for (ctx in c("math", "code", "facts", "general")) {
    for (m in MODELS) {
      a <- posteriors[[ctx]][[m]]
      b <- posteriors[[paste0(ctx, "_b")]][[m]]
      x <- seq(0, 1, length.out = 100)
      y <- dbeta(x, max(a, 0.1), max(b, 0.1))
      df <- rbind(df, data.frame(
        context = ctx, model = m, x = x, density = y,
        stringsAsFactors = FALSE
      ))
    }
  }

  p <- ggplot(df, aes(x = x, y = density, color = model)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~ context, scales = "free_y") +
    labs(title = "Beta Posterior Distributions by Context",
         subtitle = paste("After", nrow(history), "episodes"),
         x = "Success Probability", y = "Density") +
    theme_minimal() +
    scale_color_brewer(palette = "Set2")

  ggsave(output_file, p, width = 10, height = 8, dpi = 150)
  return(output_file)
}

# ─── Plot: learning curve ────────────────────────────────────

plot_learning <- function(output_file = "learning.png") {
  if (nrow(history) < 2) return(NULL)

  history$cumulative_acc <- cumsum(history$reward) / seq_len(nrow(history))

  p <- ggplot(history, aes(x = seq_len(nrow(history)), y = cumulative_acc)) +
    geom_line(color = "#2563eb", linewidth = 1) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "#999") +
    labs(title = "Cumulative Routing Accuracy",
         x = "Episode", y = "Accuracy") +
    theme_minimal()

  ggsave(output_file, p, width = 8, height = 4, dpi = 150)
  return(output_file)
}

# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  # Demo mode
  cat("=== R Bayesian Router Demo ===\n\n")

  questions <- list(
    c("Solve 2x+5=17", 0.95),
    c("What is the capital of France?", 0.9),
    c("Write Python quicksort", 0.9),
    c("Prove Lagrange theorem", 0.95),
    c("How many legs does a spider have?", 0.85)
  )

  for (i in 1:5) {
    q <- questions[[i]][1]; r <- as.numeric(questions[[i]][2])
    result <- route(q)
    learn(q, result$primary_model, r)
    cat(sprintf("[%d] %s\n    -> %s (context: %s, expected: %.3f)\n",
        i, substr(q, 1, 40), result$primary_model, result$context,
        result$expected_probs[result$primary_model]))
  }

  s <- summarize()
  cat(sprintf("\nTotal: %d episodes | Avg reward: %.3f\n", s$total_episodes, s$avg_reward))
  cat("Model usage:\n")
  print(s$models_used)

} else {
  cmd <- args[1]
  if (cmd == "--command" && length(args) >= 2) {
    subcmd <- args[2]
    if (subcmd == "route") {
      q <- paste(args[-(1:2)], collapse = " ")
      cat(toJSON(route(q), auto_unbox = TRUE, pretty = TRUE))
    } else if (subcmd == "learn") {
      # --command learn --model ds-think --reward 1.0 --question "..."
      model <- args[grep("--model", args) + 1]
      reward <- as.numeric(args[grep("--reward", args) + 1])
      q <- paste(args[-(1:which(args == "--question"))], collapse = " ")
      cat(toJSON(learn(q, model, reward), auto_unbox = TRUE))
    } else if (subcmd == "summarize") {
      cat(toJSON(summarize(), auto_unbox = TRUE, pretty = TRUE))
    } else if (subcmd == "plot") {
      output <- args[grep("--output", args) + 1]
      plot_posteriors(output)
      cat(sprintf("Saved: %s\n", output))
    }
  }
}
