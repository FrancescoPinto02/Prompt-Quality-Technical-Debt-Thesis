# ============================================================
# FOCUSED REPLICATION:
# Flesch Reading Ease + Gunning Fog Index
# Outcomes: Usefulness e Correctness
# ============================================================

# Installare solo la prima volta, se necessario
# install.packages(c("car", "MASS", "brant", "VGAM"))

library(car)
library(MASS)
library(brant)
library(VGAM)


# ============================================================
# 1. CARICAMENTO DATI
# ============================================================

data <- read.csv("olr_dataset.csv")


# ============================================================
# 2. PREPARAZIONE
# ============================================================

data_analysis <- data[, c(
  "flesch_reading_ease",
  "gunning_fog_index",
  "usefulness",
  "correctness"
)]

# Numero iniziale di osservazioni
n_initial <- nrow(data_analysis)

# Rimozione eventuali valori mancanti
data_analysis <- na.omit(data_analysis)

n_removed <- n_initial - nrow(data_analysis)

# Controllo che gli outcome siano compresi tra 0 e 4
if (!all(data_analysis$usefulness %in% 0:4) ||
    !all(data_analysis$correctness %in% 0:4)) {
  stop("Usefulness o correctness contengono valori fuori dall'intervallo 0-4.")
}

# Conversione in variabili ordinali
data_analysis$usefulness_ord <- ordered(
  data_analysis$usefulness,
  levels = 0:4
)

data_analysis$correctness_ord <- ordered(
  data_analysis$correctness,
  levels = 0:4
)


# ============================================================
# 3. MULTICOLLINEARITÀ - VIF
# ============================================================

# L'outcome usato qui non modifica il VIF:
# il VIF dipende solamente dai predittori.
vif_model <- lm(
  usefulness ~
    flesch_reading_ease +
    gunning_fog_index,
  data = data_analysis
)

vif_results <- car::vif(vif_model)


# ============================================================
# 4. PROPORTIONAL ODDS - BRANT TEST
# ============================================================

# ----- USEFULNESS -----

olr_usefulness <- MASS::polr(
  usefulness_ord ~
    flesch_reading_ease +
    gunning_fog_index,
  data = data_analysis,
  Hess = TRUE,
  method = "logistic"
)

brant_usefulness <- capture.output(
  brant::brant(olr_usefulness, by.var = TRUE)
)


# ----- CORRECTNESS -----

olr_correctness <- MASS::polr(
  correctness_ord ~
    flesch_reading_ease +
    gunning_fog_index,
  data = data_analysis,
  Hess = TRUE,
  method = "logistic"
)

brant_correctness <- capture.output(
  brant::brant(olr_correctness, by.var = TRUE)
)


# ============================================================
# 5. GENERALIZED ORDINAL LOGISTIC REGRESSION
# ============================================================


# ----- USEFULNESS -----

golr_usefulness <- VGAM::vglm(
  usefulness_ord ~
    flesch_reading_ease +
    gunning_fog_index,
  family = VGAM::cumulative(
    link = "logitlink",
    parallel = FALSE,
    reverse = TRUE
  ),
  data = data_analysis
)


# ----- CORRECTNESS -----

golr_correctness <- VGAM::vglm(
  correctness_ord ~
    flesch_reading_ease +
    gunning_fog_index,
  family = VGAM::cumulative(
    link = "logitlink",
    parallel = FALSE,
    reverse = TRUE
  ),
  data = data_analysis
)


# Coefficienti organizzati per soglia
coef_usefulness <- coef(golr_usefulness, matrix = TRUE)
coef_correctness <- coef(golr_correctness, matrix = TRUE)

# Odds Ratio dei due predittori
or_usefulness <- exp(
  coef_usefulness[
    c("flesch_reading_ease", "gunning_fog_index"),
    ,
    drop = FALSE
  ]
)

or_correctness <- exp(
  coef_correctness[
    c("flesch_reading_ease", "gunning_fog_index"),
    ,
    drop = FALSE
  ]
)


# ============================================================
# 6. SALVATAGGIO RISULTATI
# ============================================================

results_file <- "risultati_RQ1_RQ2.txt"

results <- c(
  
  "============================================================",
  "FOCUSED REPLICATION",
  "Predictors: Flesch Reading Ease + Gunning Fog Index",
  "============================================================",
  "",
  
  "DATI",
  paste("Osservazioni iniziali:", n_initial),
  paste("Osservazioni rimosse per missing:", n_removed),
  paste("Osservazioni analizzate:", nrow(data_analysis)),
  "",
  
  "Distribuzione Usefulness:",
  capture.output(print(table(data_analysis$usefulness))),
  "",
  
  "Distribuzione Correctness:",
  capture.output(print(table(data_analysis$correctness))),
  "",
  
  
  "============================================================",
  "VIF",
  "============================================================",
  capture.output(print(round(vif_results, 3))),
  "",
  
  
  "============================================================",
  "BRANT TEST - USEFULNESS",
  "============================================================",
  brant_usefulness,
  "",
  
  
  "============================================================",
  "BRANT TEST - CORRECTNESS",
  "============================================================",
  brant_correctness,
  "",
  
  
  "============================================================",
  "GENERALIZED OLR - USEFULNESS",
  "============================================================",
  capture.output(summary(golr_usefulness)),
  "",
  
  "Odds Ratios per soglia:",
  capture.output(print(round(or_usefulness, 4))),
  "",
  
  
  "============================================================",
  "GENERALIZED OLR - CORRECTNESS",
  "============================================================",
  capture.output(summary(golr_correctness)),
  "",
  
  "Odds Ratios per soglia:",
  capture.output(print(round(or_correctness, 4)))
)

writeLines(results, results_file)


# ============================================================
# 7. CONFERMA
# ============================================================

cat("\nAnalisi completata.\n")
cat("Risultati salvati in:\n")
cat(normalizePath(results_file), "\n")