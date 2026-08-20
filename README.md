# From Prompt Quality to Code Quality

This repository contains the replication package for the Master’s thesis **“From Prompt Quality to Code Quality: An Empirical Study on Developer–LLM Conversations”**.
The repository includes the scripts, prompts, configurations, and analysis utilities used to reproduce the empirical pipeline, from dataset preparation to prompt-metric extraction, ICE-Score evaluation, static analysis, and statistical modeling.

## Thesis Objective

The goal of this thesis is to study whether readability-related characteristics of developer prompts are associated with the quality of LLM-generated code.
The study is based on **CodeChat-V2.0** and focuses on English single-turn developer–LLM conversations. Prompt quality is operationalized through readability-related metrics, while generated-code quality is evaluated through ICE-Score and static-analysis findings.

## Research Questions

- **RQ1:** Are prompt readability and linguistic complexity associated with the usefulness of LLM-generated code?
- **RQ2:** Are prompt readability and linguistic complexity associated with the functional correctness of LLM-generated code?
- **RQ3:** Which readability-related prompt metrics are associated with the presence of high-severity static-analysis findings in LLM-generated code?
- **RQ4:** Which readability-related prompt metrics are associated with the number of high-severity static-analysis findings in LLM-generated code?

## Main Results

The results show that readability-related prompt metrics are associated with some dimensions of generated-code quality, but not uniformly.

- For **Usefulness** and **Functional Correctness**, Flesch Reading Ease and Gunning Fog Index are both statistically significant. More readable prompts are generally associated with better generated-code quality, while linguistic complexity shows a threshold-dependent effect.

- For internal code quality, **Gunning Fog Index** is significantly associated with the presence of at least one high-severity static-analysis finding. **Difficult Words** is significantly associated with the number of high-severity findings.

Overall, the results suggest that readability-related prompt metrics can provide useful signals for analyzing LLM-generated code quality, but they should not be interpreted as a complete measure of prompt quality.

## Repository Structure

The repository is organized around four main directories: `data`, `prompt`, `scripts`, and `tools`.

### `data/`

The `data/` directory contains the datasets and generated artifacts used throughout the empirical pipeline. It is organized to separate raw inputs, intermediate outputs, final datasets, static-analysis results, prompt metrics, ICE-Score outputs, and statistical-analysis datasets.

### `prompt/`

The `prompt/` directory contains the prompt templates used by the LLM-based annotation and evaluation steps. These prompts are stored separately from the Python code to make the pipeline easier to inspect, modify, and reproduce.
This directory includes prompts for natural-language/code separation, language annotation, task classification, topic classification, and ICE-Score evaluation.

### `scripts/`

The `scripts/` directory contains the Python scripts used to run the full replication pipeline. These scripts implement dataset filtering, prompt processing, language/task/topic annotation, prompt-metric extraction, ICE-Score computation, static-analysis execution, dataset construction, plotting, and statistical-analysis preparation.

### `tools/`

The `tools/` directory contains tool-specific configurations and auxiliary projects required by the static-analysis pipeline.
This includes, for example, ESLint configuration files for JavaScript analysis, PMD rulesets for Java analysis, and the Roslyn analyzer project used for C# analysis. Generated build artifacts, installed dependencies, and temporary working directories should not be committed to the repository.


## Requirements

This replication package requires both Python dependencies and external tools used by the preprocessing, annotation, evaluation, and static-analysis pipelines.

### Python

Use Python 3.10 or later. The Python dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main Python dependencies include libraries for dataset processing, prompt-metric extraction, LLM/API calls, plotting, and static-analysis orchestration.

### LLM API Access

Some pipeline steps require access to an OpenAI-compatible chat completions endpoint. This includes natural-language/code splitting, language annotation, task classification, topic classification, and ICE-Score computation.

The scripts can be configured to use either a local endpoint, such as LM Studio, or a remote provider exposing an OpenAI-compatible API.

If an API key is required, set it through an environment variable:

```bash
export OPENAI_API_KEY="your_api_key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

### JavaScript / ESLint

JavaScript static analysis requires Node.js and npm. The ESLint configuration and Node dependencies are stored in:

```text
tools/eslint_env/
```

Install the JavaScript dependencies with:

```bash
cd tools/eslint_env
npm install
cd ../..
```


### Java / PMD

Java static analysis requires a Java runtime or JDK and PMD. The PMD rulesets used by the replication package are stored in:

```text
tools/pmd_rulesets/
```

PMD itself should be installed separately and made available from the command line, or its path should be configured in the corresponding static-analysis script.

### C++ / Cppcheck

C++ static analysis requires Cppcheck. Cppcheck should be installed separately and available from the command line, or its executable path should be configured in the corresponding static-analysis script.

### C# / Roslyn

C# static analysis requires the .NET SDK.
The Roslyn analyzer project is stored in:

```text
tools/roslyn_analyzer/RoslynSnippetAnalyzer/
```

Before running the C# static-analysis script, restore and build the project:

```bash
dotnet restore tools/roslyn_analyzer/RoslynSnippetAnalyzer/RoslynSnippetAnalyzer.csproj
dotnet build tools/roslyn_analyzer/RoslynSnippetAnalyzer/RoslynSnippetAnalyzer.csproj
```

Do not commit generated `bin/` or `obj/` directories.

###  R Environment

The statistical-analysis scripts are run in R, install the R packages required by those scripts before reproducing the regression analyses. These may include packages for data loading, regression modeling, model diagnostics, and result reporting.


## Main Reproduction Steps

This section summarizes the main steps required to reproduce the empirical pipeline. The exact execution order may be adapted depending on whether intermediate datasets are regenerated from scratch or reused from the replication package.


### 1. Filter the raw dataset

Run the dataset filtering script to retain English single-turn conversations and remove duplicated or near-duplicated initial prompts:

```bash
python scripts/dataset/filter_raw_dataset.py
```

This step produces the filtered dataset in:

```text
data/filtered/
```

and saves filtering reports in:

```text
data/filtered/report/
```

### 2. Process user prompts

Run the processing script to split each user prompt into natural-language and code-like parts and to validate the language of the natural-language request:

```bash
python scripts/dataset/run_full_record_processing.py
```

This step produces JSONL files in:

```text
data/processed/v1/
```

Each output record contains the original prompt, the extracted natural-language text, the extracted code-like text, the code-inclusion flag, and the detected prompt language.


### 3. Run task and topic classification

Run the task and topic classification scripts:

```bash
python scripts/intent_classification/run_task_classification.py
python scripts/intent_classification/run_topic_classification.py
```


### 4. Build the final dataset

Build the final cleaned dataset by combining the filtered conversations with the processed prompt annotations:

```bash
python scripts/dataset/build_final_dataset.py
```


### 5. Compute prompt-oriented metrics

Compute readability-related prompt metrics on the natural-language portion of each prompt:

```bash
python scripts/prompt_metrics/calculate_prompt_oriented_metrics.py
```

This step produces:

```text
data/prompt_metrics/prompt_metrics.csv
```

The main metrics are Flesch Reading Ease, Gunning Fog Index, Difficult Words, Yule’s K, and Number of Sentences.

### 6. Compute ICE-Score evaluations

Run ICE-Score evaluation for Usefulness and Functional Correctness:

```bash
python scripts/ice_score/run_ice_usefulness.py
python scripts/ice_score/run_ice_correctness.py
python scripts/ice_score/merge_ice_scores.py
```

The ICE-Score prompts are stored in:

```text
prompt/ice_score/
```

The resulting scores are saved in:

```text
data/ice_score/
```

### 7. Run static analysis

Run the language-specific static-analysis scripts for the selected programming languages:

```bash
python scripts/static_analysis/run_pylint_analysis.py
python scripts/static_analysis/run_eslint_analysis.py
python scripts/static_analysis/run_cppcheck_analysis.py
python scripts/static_analysis/run_pmd_analysis.py
python scripts/static_analysis/run_roslyn_analysis.py
```

These scripts analyze generated code snippets in Python, JavaScript, C++, Java, and C#.

The outputs are stored in:

```text
data/static_analysis/v1/
```

The external tools required for this step are configured through files in:

```text
tools/
```

### 8. Build analysis-ready datasets

Build the datasets used for the statistical analyses.

For ICE-Score analyses:

```bash
python scripts/analysis/build_olr_dataset.py
```

For the pooled static-analysis dataset:

```bash
python scripts/analysis/build_pooled_static_analysis_dataset.py
```

The resulting datasets are saved in:

```text
data/analysis/
```

### 9. Run statistical analyses

Run the statistical-analysis scripts to reproduce the regression models used in the thesis.

The main models are:

- Generalized Ordinal Logistic Regression for Usefulness.
- Generalized Ordinal Logistic Regression for Functional Correctness.
- Binary Logistic Regression for the presence of high-severity static-analysis findings.
- Negative Binomial Regression for the number of high-severity static-analysis findings.

The R scripts used for the analysis are stored in the `scripts/R-analysis` directory

### Notes

Some steps require an OpenAI-compatible LLM endpoint. Before running them, check the model name, API base URL, and API key configuration in the corresponding scripts.

## Future Work

Future work can extend this study in several directions:

- Multi-turn conversations: extend the analysis beyond single-turn interactions to study how iterative prompt refinements affect generated-code quality.
- Structural prompt metrics: investigate scalable ways to include prompt patterns, task framing, and other structural characteristics of prompts.
- Counterfactual analysis: modify prompts in a controlled way to assess whether changes in readability-related metrics lead to measurable changes in generated-code quality.
- Additional languages and tools: extend the static-analysis pipeline to other programming languages and static-analysis ecosystems.
- Prompt feedback tools: explore the development of tools that use prompt-oriented metrics to provide developers with feedback before submitting prompts to LLM-based coding assistants.