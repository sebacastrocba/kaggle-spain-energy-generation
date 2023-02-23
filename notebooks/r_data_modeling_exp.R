# MODELING

# Time Series ML
library(tidymodels)
library(modeltime)

# Core
library(tidyverse)
library(lubridate)
library(timetk)

# Timing & Parallel Processing
library(tictoc)
library(future)
library(doFuture)

# Parallel computing
registerDoFuture()
n_cores <- parallel::detectCores()
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

# Data loading
data <- read_csv("../data/processed/reservoir_cleaned_data.csv")
data %>% head()

# Data Transformation
reservoir_transformed_tbl <- data %>%
  
  # Preprocess Target
  mutate(generation_trans = log_interval_vec(generation_cleaned, limit_lower = 0, offset = 1)) %>%
  mutate(generation_trans = standardize_vec(generation_cleaned)) %>%
  select(-generation_cleaned)

# Save parameters
gen_mean <- 2605.70790454515
gen_std <- 1835.09500627984
limit_lower <- 0
offset <- 1

# Step 1: Create Full dataset
horizon <- 24
lag_period <- 24
rolling_periods <- c(24, 48, 72)

data_prepared_full_tbl <- reservoir_transformed_tbl %>%
  
  # Add future window
  bind_rows(
    future_frame(.data = ., .date_var = timestamp, .length_out = horizon)
  ) %>%
  
  # Add Autocorrelated Lags
  tk_augment_lags(generation_trans, .lags = lag_period) %>%
  
  # Add rolling features
  tk_augment_slidify(
    .value   = generation_trans_lag24,
    .f       = mean, 
    .period  = rolling_periods,
    .align   = "center",
    .partial = TRUE
  ) %>%
  
  # Format Columns
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))

data_prepared_full_tbl %>%
  pivot_longer(-timestamp) %>%
  plot_time_series(timestamp, value, name, .smooth = FALSE)

# 2.0 STEP 2 - SEPARATE INTO MODELING & FORECAST DATA ----

data_prepared_full_tbl %>% tail(57)

data_prepared_tbl <- data_prepared_full_tbl %>%
  filter(!is.na(generation_trans)) %>% 
  drop_na()

data_prepared_tbl

forecast_tbl <- data_prepared_full_tbl %>%
  filter(is.na(generation_trans))

forecast_tbl

# 3.0 TRAIN/TEST (MODEL DATASET) ----

splits <- time_series_split(data_prepared_tbl, assess = horizon, cumulative = TRUE)

splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(timestamp, generation_trans)

# 4.0 RECIPES ----

recipe_spec_base <- recipe(generation_trans ~ ., data = training(splits)) %>%
  
  # Time Series Signature
  step_timeseries_signature(timestamp) %>%
  step_rm(matches("(iso)|(xts)|(minute)|(second)|(am.pm)")) %>%
  
  # Standardization
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  
  # Dummy Encoding (One Hot Encoding)
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  
  # Interaction
  step_interact(~ matches("week2") * matches("wday.lbl")) %>%
  
  # Fourier
  step_fourier(timestamp, period = c(12, 24, 36, 48, 50), K = 2)

recipe_spec_base %>% prep() %>% juice() %>% glimpse()

# 5.0 SPLINE MODEL ----

# * LM Model Spec ----

model_spec_lm <- linear_reg() %>%
  set_engine("lm")

# * Spline Recipe Spec ----

recipe_spec_1 <- recipe_spec_base %>%
  step_rm(timestamp) %>%
  step_ns(ends_with("index.num"), deg_free = 2) %>%
  step_rm(starts_with("lag_"))

recipe_spec_1 %>% prep() %>% juice() %>% glimpse()

# * Spline Workflow  ----

workflow_fit_lm_1_spline <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(recipe_spec_1) %>%
  fit(training(splits))

workflow_fit_lm_1_spline

workflow_fit_lm_1_spline %>% 
  extract_fit_parsnip() %>%
  pluck("fit") %>%
  summary()

# 6.0 MODELTIME  ----

calibration_tbl <- modeltime_table(
  workflow_fit_lm_1_spline
) %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
  modeltime_forecast(new_data    = testing(splits), 
                     actual_data = data_prepared_tbl) %>%
  plot_modeltime_forecast()

calibration_tbl %>% modeltime_accuracy()

# 7.0 LAG MODEL ----

# * Lag Recipe ----

recipe_spec_base %>% prep() %>% juice() %>% glimpse()

recipe_spec_2 <- recipe_spec_base %>%
  step_rm(timestamp) %>%
  step_naomit(starts_with("lag_"))

recipe_spec_2 %>% prep() %>% juice() %>% glimpse()

# * Lag Workflow ----

workflow_fit_lm_2_lag <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(recipe_spec_2) %>%
  fit(training(splits))

workflow_fit_lm_2_lag

workflow_fit_lm_2_lag %>% extract_fit_parsnip() %>% pluck("fit") %>% summary()

# * Compare with Modeltime -----

calibration_tbl <- modeltime_table(
  workflow_fit_lm_1_spline,
  workflow_fit_lm_2_lag
) %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
  modeltime_forecast(new_data    = testing(splits), 
                     actual_data = data_prepared_tbl) %>%
  plot_modeltime_forecast()

calibration_tbl %>%
  modeltime_accuracy()

# * Parsnip Model (ARIMA)  ----

model_fit_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(generation_trans ~ timestamp, data = training(splits))

# * Workflow (ARIMA + Date Features) ----

model_spec_arima <- arima_reg() %>%
  set_engine("auto_arima")

recipe_spec_fourier <- recipe(generation_trans ~ timestamp, data = training(splits)) %>%
  step_fourier(timestamp, period = c(12, 24, 48, 60, 72), K = 1) 

recipe_spec_fourier %>% prep() %>% juice() %>% glimpse()

workflow_fit_arima <- workflow() %>%
  add_recipe(recipe_spec_fourier) %>%
  add_model(model_spec_arima) %>%
  fit(training(splits))

workflow_fit_arima

# * Workflow (GLMNET + XREGS) ----

recipe_spec_2

recipe_spec_2 %>% prep() %>% juice()

model_spec_glmnet <- linear_reg(
  penalty = 0.1,
  mixture = 0.5
) %>%
  set_engine("glmnet")

workflow_fit_glmnet <- workflow() %>%
  add_recipe(recipe_spec_2) %>%
  add_model(model_spec_glmnet) %>%
  fit(training(splits))

workflow_fit_glmnet

# 2.0 MODELTIME TABLE ----
# - Organize

model_tbl <- modeltime_table(
  model_fit_arima,
  workflow_fit_arima,
  workflow_fit_glmnet,
  workflow_fit_lm_1_spline,
  workflow_fit_lm_2_lag
) %>%
  update_model_description(3, "GLMNET - Lag Recipe") %>% 
  update_model_description(5, "LM - Lag Recipe")

calibration_tbl <- model_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
  slice(1) %>%
  unnest(.calibration_data)

calibration_tbl %>%
  modeltime_accuracy()
