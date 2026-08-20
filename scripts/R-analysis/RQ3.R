# ============================================================
# Regressione logistica binaria + analisi di sensibilità spline
# Leggibilità prompt -> presenza errori Pylint
# ============================================================

# 1. Caricare Dataset
# ------------------------------------------------------------

dataset_path <- "static_analysis_dataset.csv"

df <- read.csv(dataset_path, stringsAsFactors = FALSE)

required_packages <- c("car", "splines")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 2. Preparare e trasformare i dati
# ------------------------------------------------------------

vars_indipendenti <- c(
  "flesch_reading_ease",
  "gunning_fog_index",
  "difficult_words",
  "yules_k",
  "number_of_sentences"
)

var_dipendente <- "has_severe_static_analysis_finding"

df_model <- df[, c(vars_indipendenti, var_dipendente)]

# Conversione robusta della variabile dipendente True/False -> 1/0
df_model$has_severe_static_analysis_finding <- tolower(as.character(df_model$has_severe_static_analysis_finding))

df_model$has_severe_static_analysis_finding <- ifelse(
  df_model$has_severe_static_analysis_finding %in% c("true", "1", "yes", "y"),
  1,
  ifelse(df_model$has_severe_static_analysis_finding %in% c("false", "0", "no", "n"), 0, NA)
)

# Conversione delle variabili indipendenti in numeriche
df_model[vars_indipendenti] <- lapply(
  df_model[vars_indipendenti],
  function(x) as.numeric(as.character(x))
)

# Rimozione righe con NA dopo le conversioni
df_model <- na.omit(df_model)

# Controllo che restino osservazioni
if (nrow(df_model) == 0) {
  stop("Errore: dopo la rimozione degli NA non resta nessuna osservazione.")
}

# Controllo che la variabile dipendente sia binaria
if (!all(df_model$has_severe_static_analysis_finding %in% c(0, 1))) {
  stop("Errore: la variabile dipendente non è codificata correttamente come 0/1.")
}

# Standardizzazione predittori per modello classico
df_model_scaled <- df_model
df_model_scaled[vars_indipendenti] <- scale(df_model_scaled[vars_indipendenti])

# Formula modello classico
formula_logit <- as.formula(
  paste(var_dipendente, "~", paste(vars_indipendenti, collapse = " + "))
)

# 3. Controllo VIF
# ------------------------------------------------------------

modello_vif <- lm(formula_logit, data = df_model_scaled)

vif_results <- vif(modello_vif)

# 4. Fitting modello logistico classico
# ------------------------------------------------------------

modello_logistico <- glm(
  formula_logit,
  data = df_model_scaled,
  family = binomial(link = "logit")
)

summary_modello <- summary(modello_logistico)

# Odds Ratio e intervalli di confidenza modello classico
odds_ratios <- exp(coef(modello_logistico))
conf_int <- exp(confint(modello_logistico))

risultati_or <- data.frame(
  Variabile = names(odds_ratios),
  Odds_Ratio = odds_ratios,
  CI_2.5 = conf_int[, 1],
  CI_97.5 = conf_int[, 2],
  row.names = NULL
)

# 5. Controllo Logit Lineari - Box-Tidwell
# ------------------------------------------------------------
# Box-Tidwell: x * log(x)
# Richiede valori strettamente positivi.

df_bt <- df_model

for (v in vars_indipendenti) {
  min_val <- min(df_bt[[v]], na.rm = TRUE)
  
  if (min_val <= 0) {
    df_bt[[v]] <- df_bt[[v]] + abs(min_val) + 0.001
  }
}

for (v in vars_indipendenti) {
  df_bt[[paste0(v, "_log")]] <- df_bt[[v]] * log(df_bt[[v]])
}

formula_bt <- as.formula(
  paste(
    var_dipendente,
    "~",
    paste(
      c(vars_indipendenti, paste0(vars_indipendenti, "_log")),
      collapse = " + "
    )
  )
)

modello_box_tidwell <- glm(
  formula_bt,
  data = df_bt,
  family = binomial(link = "logit")
)

summary_box_tidwell <- summary(modello_box_tidwell)

# Identificazione automatica delle variabili che violano la linearità del logit
coef_bt <- summary_box_tidwell$coefficients

termini_log <- paste0(vars_indipendenti, "_log")

p_values_bt <- coef_bt[termini_log, "Pr(>|z|)"]

variabili_violano_logit <- vars_indipendenti[p_values_bt < 0.05]

# 6. Modello spline per analisi di sensibilità
# ------------------------------------------------------------
# Applica spline solo alle variabili che violano la linearità del logit.
# Le altre restano lineari.

df_spline <- df_model_scaled

modello_spline <- NULL
summary_modello_spline <- NULL
confronto_modelli <- NULL
confronto_spline_senza_variabili <- list()
formula_spline <- NULL

if (length(variabili_violano_logit) > 0) {
  
  termini_modello_spline <- c()
  
  for (v in vars_indipendenti) {
    if (v %in% variabili_violano_logit) {
      termini_modello_spline <- c(
        termini_modello_spline,
        paste0("ns(", v, ", df = 3)")
      )
    } else {
      termini_modello_spline <- c(termini_modello_spline, v)
    }
  }
  
  formula_spline <- as.formula(
    paste(
      var_dipendente,
      "~",
      paste(termini_modello_spline, collapse = " + ")
    )
  )
  
  modello_spline <- glm(
    formula_spline,
    data = df_spline,
    family = binomial(link = "logit")
  )
  
  summary_modello_spline <- summary(modello_spline)
  
  # Confronto modello classico vs modello spline
  confronto_modelli <- anova(
    modello_logistico,
    modello_spline,
    test = "Chisq"
  )
  
  # Verifica della significatività complessiva delle variabili spline
  # tramite confronto tra modello spline completo e modello senza la variabile spline.
  
  for (var_spline in variabili_violano_logit) {
    
    termini_senza_var <- c()
    
    for (v in vars_indipendenti) {
      if (v == var_spline) {
        next
      } else if (v %in% variabili_violano_logit) {
        termini_senza_var <- c(
          termini_senza_var,
          paste0("ns(", v, ", df = 3)")
        )
      } else {
        termini_senza_var <- c(termini_senza_var, v)
      }
    }
    
    formula_senza_var <- as.formula(
      paste(
        var_dipendente,
        "~",
        paste(termini_senza_var, collapse = " + ")
      )
    )
    
    modello_senza_var <- glm(
      formula_senza_var,
      data = df_spline,
      family = binomial(link = "logit")
    )
    
    confronto_spline_senza_variabili[[var_spline]] <- anova(
      modello_senza_var,
      modello_spline,
      test = "Chisq"
    )
  }
}

# 7. Salvataggio risultati in file txt
# ------------------------------------------------------------

output_file <- "risultati_RQ3.txt"

sink(output_file)

cat("============================================================\n")
cat("ANALISI REGRESSIONE LOGISTICA BINARIA\n")
cat("CON ANALISI DI SENSIBILITA' MEDIANTE SPLINE\n")
cat("============================================================\n\n")

cat("Variabile dipendente:\n")
cat(var_dipendente, "\n\n")

cat("Variabili indipendenti:\n")
cat(paste(vars_indipendenti, collapse = ", "), "\n\n")

cat("Numero osservazioni usate nel modello:\n")
cat(nrow(df_model_scaled), "\n\n")

cat("Distribuzione variabile dipendente:\n")
print(table(df_model_scaled$has_severe_static_analysis_finding))
cat("\n")

cat("============================================================\n")
cat("CONTROLLO VIF\n")
cat("============================================================\n\n")
print(vif_results)

cat("\nInterpretazione VIF:\n")
cat("VIF > 5: possibile multicollinearità.\n")
cat("VIF > 10: multicollinearità severa.\n\n")

cat("============================================================\n")
cat("MODELLO LOGISTICO CLASSICO\n")
cat("============================================================\n\n")
print(summary_modello)

cat("\n============================================================\n")
cat("ODDS RATIO E INTERVALLI DI CONFIDENZA - MODELLO CLASSICO\n")
cat("============================================================\n\n")
print(risultati_or)

cat("\nInterpretazione Odds Ratio:\n")
cat("OR > 1 indica aumento degli odds di errore\n")
cat("OR < 1 indica riduzione degli odds di errore\n")
cat("Gli OR si riferiscono a un aumento di una deviazione standard del predittore.\n\n")

cat("============================================================\n")
cat("CONTROLLO LOGIT LINEARI - BOX-TIDWELL\n")
cat("============================================================\n\n")
print(summary_box_tidwell)

cat("\nP-value dei termini Box-Tidwell:\n\n")
print(p_values_bt)

cat("\nVariabili che violano la linearità del logit, soglia p < 0.05:\n\n")

if (length(variabili_violano_logit) > 0) {
  print(variabili_violano_logit)
} else {
  cat("Nessuna variabile viola l'assunzione di linearità del logit.\n")
}

cat("\nInterpretazione Box-Tidwell:\n")
cat("I termini con suffisso '_log' testano la linearità del logit.\n")
cat("Se un termine '_log' è significativo, l'assunzione di linearità del logit può essere violata.\n\n")

cat("============================================================\n")
cat("ANALISI DI SENSIBILITA' CON SPLINE\n")
cat("============================================================\n\n")

if (length(variabili_violano_logit) > 0) {
  
  cat("Le seguenti variabili sono state modellate con natural spline:\n")
  print(variabili_violano_logit)
  cat("\n")
  
  cat("Formula modello spline:\n")
  print(formula_spline)
  cat("\n")
  
  cat("============================================================\n")
  cat("MODELLO LOGISTICO CON SPLINE\n")
  cat("============================================================\n\n")
  print(summary_modello_spline)
  
  cat("\n============================================================\n")
  cat("CONFRONTO MODELLO CLASSICO VS MODELLO SPLINE\n")
  cat("============================================================\n\n")
  print(confronto_modelli)
  
  cat("\nInterpretazione confronto modello classico vs spline:\n")
  cat("Un p-value significativo indica che il modello spline migliora significativamente il fit.\n")
  cat("Un AIC più basso indica migliore adattamento relativo.\n\n")
  
  cat("AIC modello classico:\n")
  print(AIC(modello_logistico))
  cat("\n")
  
  cat("AIC modello spline:\n")
  print(AIC(modello_spline))
  cat("\n")
  
  cat("============================================================\n")
  cat("SIGNIFICATIVITA' COMPLESSIVA DELLE VARIABILI SPLINE\n")
  cat("============================================================\n\n")
  
  for (var_spline in names(confronto_spline_senza_variabili)) {
    cat("Confronto modello senza", var_spline, "vs modello spline completo:\n\n")
    print(confronto_spline_senza_variabili[[var_spline]])
    cat("\n")
    
    cat("Interpretazione:\n")
    cat("Se il p-value è significativo, la variabile", var_spline,
        "rimane associata significativamente alla presenza di errore anche modellandola con spline.\n\n")
  }
  
} else {
  
  cat("Nessuna spline applicata perché nessuna variabile ha violato la linearità del logit.\n")
}

cat("============================================================\n")
cat("CONCLUSIONE OPERATIVA\n")
cat("============================================================\n\n")

cat("Il modello logistico classico valuta l'associazione lineare tra le metriche di leggibilità e la presenza di errori Pylint.\n")
cat("Il test Box-Tidwell controlla se i predittori continui rispettano l'assunzione di linearità del logit.\n")
cat("Quando una variabile viola tale assunzione, il modello con spline viene usato come analisi di sensibilità.\n")
cat("Se la variabile rimane significativa nel confronto likelihood ratio, l'associazione è più robusta.\n")
cat("Se invece perde significatività, il risultato del modello classico va interpretato con cautela.\n\n")

cat("============================================================\n")
cat("FINE ANALISI\n")
cat("============================================================\n")

sink()

cat("Analisi completata. Risultati salvati in:", output_file, "\n")