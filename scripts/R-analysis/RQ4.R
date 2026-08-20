# ============================================================
# Count model con offset per linee di codice
# Outcome: severe_static_analysis_issue_count
# Offset: generated_code_line_count
# ============================================================

# ------------------------------------------------------------
# 1. Pacchetti
# ------------------------------------------------------------

required_packages <- c("MASS")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ------------------------------------------------------------
# 2. Caricamento dataset
# ------------------------------------------------------------

dataset_path <- "static_analysis_dataset.csv"

if (!file.exists(dataset_path)) {
  stop("Dataset non trovato: ", dataset_path)
}

df <- read.csv(dataset_path, stringsAsFactors = FALSE)

# ------------------------------------------------------------
# 3. Definizione variabili
# ------------------------------------------------------------

var_dipendente <- "severe_static_analysis_issue_count"

var_offset <- "generated_code_line_count"

vars_indipendenti <- c(
  "flesch_reading_ease",
  "gunning_fog_index",
  "difficult_words",
  "yules_k",
  "number_of_sentences"
)

colonne_richieste <- c(
  var_dipendente,
  var_offset,
  vars_indipendenti
)

colonne_mancanti <- setdiff(colonne_richieste, names(df))

if (length(colonne_mancanti) > 0) {
  stop(
    "Colonne mancanti nel dataset: ",
    paste(colonne_mancanti, collapse = ", ")
  )
}

# ------------------------------------------------------------
# 4. Preparazione dati
# ------------------------------------------------------------

df_model <- df[, colonne_richieste]

# Conversione outcome, offset e predittori in numerici
df_model[[var_dipendente]] <- as.numeric(as.character(df_model[[var_dipendente]]))
df_model[[var_offset]] <- as.numeric(as.character(df_model[[var_offset]]))

df_model[vars_indipendenti] <- lapply(
  df_model[vars_indipendenti],
  function(x) as.numeric(as.character(x))
)

# Rimozione NA
n_before <- nrow(df_model)

df_model <- na.omit(df_model)

n_after_na <- nrow(df_model)

# L'offset deve essere strettamente positivo, perché si usa log(linee di codice)
df_model <- df_model[df_model[[var_offset]] > 0, ]

n_after_offset <- nrow(df_model)

cat("Osservazioni iniziali:", n_before, "\n")
cat("Osservazioni dopo rimozione NA:", n_after_na, "\n")
cat("Osservazioni usate nel modello:", n_after_offset, "\n")
cat("Righe rimosse per NA:", n_before - n_after_na, "\n")
cat("Righe rimosse per generated_code_line_count <= 0:", n_after_na - n_after_offset, "\n\n")

# Controllo outcome: conteggio intero non negativo
if (any(df_model[[var_dipendente]] < 0)) {
  stop("La variabile dipendente contiene valori negativi.")
}

if (any(df_model[[var_dipendente]] != floor(df_model[[var_dipendente]]))) {
  stop("La variabile dipendente contiene valori non interi.")
}

# Standardizzazione solo dei predittori
# NON standardizzare generated_code_line_count perché entra come offset
df_model_scaled <- df_model

df_model_scaled[vars_indipendenti] <- scale(df_model_scaled[vars_indipendenti])

# Formula con offset
formula_count_offset <- as.formula(
  paste(
    var_dipendente,
    "~",
    paste(vars_indipendenti, collapse = " + "),
    "+ offset(log(", var_offset, "))"
  )
)

# ------------------------------------------------------------
# 5. Statistiche descrittive essenziali
# ------------------------------------------------------------

cat("============================================================\n")
cat("STATISTICHE DESCRITTIVE\n")
cat("============================================================\n\n")

cat("Outcome:", var_dipendente, "\n")
print(summary(df_model[[var_dipendente]]))

cat("\nMedia outcome:", mean(df_model[[var_dipendente]]), "\n")
cat("Varianza outcome:", var(df_model[[var_dipendente]]), "\n")
cat("Percentuale zeri outcome:", mean(df_model[[var_dipendente]] == 0) * 100, "%\n\n")

cat("Offset:", var_offset, "\n")
print(summary(df_model[[var_offset]]))

cat("\n")

# ------------------------------------------------------------
# 6. Modello Poisson con offset
# ------------------------------------------------------------

modello_poisson_offset <- glm(
  formula_count_offset,
  data = df_model_scaled,
  family = poisson(link = "log")
)

cat("============================================================\n")
cat("MODELLO POISSON CON OFFSET\n")
cat("============================================================\n\n")

print(summary(modello_poisson_offset))

# Funzione per calcolare IRR e CI Wald
make_irr_table <- function(model) {
  beta <- coef(model)
  se <- sqrt(diag(vcov(model)))
  
  irr <- exp(beta)
  ci_low <- exp(beta - 1.96 * se)
  ci_high <- exp(beta + 1.96 * se)
  
  data.frame(
    Variabile = names(beta),
    IRR = irr,
    CI_2.5 = ci_low,
    CI_97.5 = ci_high,
    row.names = NULL
  )
}

risultati_poisson <- make_irr_table(modello_poisson_offset)

cat("\nIRR modello Poisson con offset:\n")
print(risultati_poisson)

# ------------------------------------------------------------
# 7. Test overdispersione
# ------------------------------------------------------------

pearson_chisq <- sum(residuals(modello_poisson_offset, type = "pearson")^2)

dispersion_ratio <- pearson_chisq / modello_poisson_offset$df.residual

p_overdispersion <- pchisq(
  pearson_chisq,
  df = modello_poisson_offset$df.residual,
  lower.tail = FALSE
)

cat("\n============================================================\n")
cat("TEST OVERDISPERSIONE\n")
cat("============================================================\n\n")

cat("Pearson Chi-square:", pearson_chisq, "\n")
cat("Gradi di libertà residui:", modello_poisson_offset$df.residual, "\n")
cat("Dispersion ratio:", dispersion_ratio, "\n")
cat("P-value overdispersione:", p_overdispersion, "\n\n")

cat("Interpretazione:\n")
cat("- Dispersion ratio circa 1: Poisson plausibile.\n")
cat("- Dispersion ratio > 1.5: possibile overdispersione.\n")
cat("- Dispersion ratio > 2: overdispersione rilevante; meglio Negative Binomial.\n\n")

# ------------------------------------------------------------
# 8. Negative Binomial con offset se necessario
# ------------------------------------------------------------

usa_negative_binomial <- dispersion_ratio > 1.5

if (usa_negative_binomial) {
  
  cat("============================================================\n")
  cat("OVERDISPERSIONE RILEVATA: STIMO NEGATIVE BINOMIAL CON OFFSET\n")
  cat("============================================================\n\n")
  
  modello_nb_offset <- glm.nb(
    formula_count_offset,
    data = df_model_scaled
  )
  
  print(summary(modello_nb_offset))
  
  risultati_nb <- make_irr_table(modello_nb_offset)
  
  cat("\nIRR modello Negative Binomial con offset:\n")
  print(risultati_nb)
  
  cat("\nConfronto AIC:\n")
  cat("AIC Poisson con offset:", AIC(modello_poisson_offset), "\n")
  cat("AIC Negative Binomial con offset:", AIC(modello_nb_offset), "\n\n")
  
  modello_finale <- modello_nb_offset
  risultati_finali <- risultati_nb
  nome_modello_finale <- "Negative Binomial con offset"
  
} else {
  
  cat("============================================================\n")
  cat("NESSUNA OVERDISPERSIONE RILEVANTE: USO POISSON CON OFFSET\n")
  cat("============================================================\n\n")
  
  modello_finale <- modello_poisson_offset
  risultati_finali <- risultati_poisson
  nome_modello_finale <- "Poisson con offset"
}

# ------------------------------------------------------------
# 9. Salvataggio risultati essenziali
# ------------------------------------------------------------

output_file <- "risultati_RQ4.txt"

sink(output_file)

cat("============================================================\n")
cat("COUNT MODEL CON OFFSET PER LINEE DI CODICE\n")
cat("============================================================\n\n")

cat("Outcome:\n")
cat(var_dipendente, "\n\n")

cat("Offset / esposizione:\n")
cat(var_offset, "\n\n")

cat("Predittori:\n")
cat(paste(vars_indipendenti, collapse = ", "), "\n\n")

cat("Osservazioni iniziali:\n")
cat(n_before, "\n\n")

cat("Osservazioni dopo rimozione NA:\n")
cat(n_after_na, "\n\n")

cat("Osservazioni usate nel modello:\n")
cat(n_after_offset, "\n\n")

cat("Righe rimosse per NA:\n")
cat(n_before - n_after_na, "\n\n")

cat("Righe rimosse per generated_code_line_count <= 0:\n")
cat(n_after_na - n_after_offset, "\n\n")

cat("============================================================\n")
cat("STATISTICHE OUTCOME\n")
cat("============================================================\n\n")

print(summary(df_model[[var_dipendente]]))

cat("\nMedia outcome:\n")
print(mean(df_model[[var_dipendente]]))

cat("\nVarianza outcome:\n")
print(var(df_model[[var_dipendente]]))

cat("\nPercentuale zeri outcome:\n")
print(mean(df_model[[var_dipendente]] == 0) * 100)

cat("\n\n============================================================\n")
cat("STATISTICHE OFFSET\n")
cat("============================================================\n\n")

print(summary(df_model[[var_offset]]))

cat("\n============================================================\n")
cat("MODELLO POISSON CON OFFSET\n")
cat("============================================================\n\n")

print(summary(modello_poisson_offset))

cat("\nIRR Poisson con offset:\n")
print(risultati_poisson)

cat("\n============================================================\n")
cat("TEST OVERDISPERSIONE\n")
cat("============================================================\n\n")

cat("Pearson Chi-square:\n")
print(pearson_chisq)

cat("\nDispersion ratio:\n")
print(dispersion_ratio)

cat("\nP-value overdispersione:\n")
print(p_overdispersion)

cat("\nInterpretazione:\n")
cat("Se il dispersion ratio è molto maggiore di 1, la Poisson è inadeguata.\n")
cat("In presenza di overdispersione rilevante, è preferibile usare Negative Binomial.\n\n")

if (usa_negative_binomial) {
  
  cat("============================================================\n")
  cat("MODELLO NEGATIVE BINOMIAL CON OFFSET\n")
  cat("============================================================\n\n")
  
  print(summary(modello_nb_offset))
  
  cat("\nIRR Negative Binomial con offset:\n")
  print(risultati_nb)
  
  cat("\nAIC Poisson con offset:\n")
  print(AIC(modello_poisson_offset))
  
  cat("\nAIC Negative Binomial con offset:\n")
  print(AIC(modello_nb_offset))
}

cat("\n============================================================\n")
cat("MODELLO FINALE SCELTO\n")
cat("============================================================\n\n")

cat("Modello finale:", nome_modello_finale, "\n\n")

cat("Risultati finali in termini di IRR:\n")
print(risultati_finali)

cat("\nNota interpretativa:\n")
cat("Il modello usa offset(log(generated_code_line_count)).\n")
cat("Quindi gli IRR si riferiscono al tasso atteso di issue severe per linea di codice.\n")
cat("Poiché i predittori sono standardizzati, ogni IRR si riferisce a un aumento di una deviazione standard del predittore.\n")
cat("IRR > 1 indica aumento del tasso atteso di issue severe per linea di codice.\n")
cat("IRR < 1 indica riduzione del tasso atteso di issue severe per linea di codice.\n")

sink()

cat("Analisi completata. Risultati salvati in:", output_file, "\n")