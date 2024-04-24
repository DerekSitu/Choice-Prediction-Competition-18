# SETUP: load packages, set working directory ----------------------------------------------------------------------------------- #
using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
using Plots

addprocs(7)
@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
end

using Flux
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

dir=@__DIR__
cd(dir)
# ============================================================================================================================== #


# OPTION ----------------------------------------------------------------------------------------------------------------------- #
attention_and_risk = 1
# 0 - no attention or risk features,
# 1 - with attention and risk features
# ============================================================================================================================== #


# Function to load and prepare the datasets
function prepare_data(csv_path)
    dataset = CSV.read(csv_path, DataFrame)
    labels = dataset.B
    features = select(dataset, Not(:SubjID, :B, :Location, :Condition))
    # - Including the analogue of "subject fixed effects" seemed intuitive but did not yield good predictions.
    # - B, the outcome, is excluded for obvious reasons.
    # - Through trial and error I found location and condition to not be good predictors.

    # DATA CLEANING ----------------------------------------------------------------------------------------------------- #
    #= # Change the values of features.Location from "Rehovot" and "Technion" to "0" and "1", convert the column to integer 
    features.Location = map(x -> x == "Rehovot" ? 0 : 1, features.Location) =#

    # Change the values of features.Gender from "M" and "F" to "0" and "1", then convert the column to integer
    features.Gender = map(x -> x == "M" ? 0 : 1, features.Gender)

    #= # Change the values of features.Condition from "ByProb" and "ByFB" to "0" and "1", then convert the column to integer
    features.Condition = map(x -> x == "ByProb" ? 0 : 1, features.Condition) =#
    
    # Create dummy variables for each value of LotShapeA
    for value in unique(features.LotShapeA)
        features[!, Symbol("LotShapeA_$value")] = map(x -> x == value ? 1 : 0, features.LotShapeA)
    end
    # Create dummy variables for each value of LotShapeB
    for value in unique(features.LotShapeB)
        features[!, Symbol("LotShapeB_$value")] = map(x -> x == value ? 1 : 0, features.LotShapeB)
    end
    # Remove LotShapeA and LotShapeB columns
    features = select(features, Not(:LotShapeA, :LotShapeB))
    # Change the values of features.Button from "L" and "R" to "0" and "1", then convert the column to integer
    features.Button = map(x -> x == "L" ? 0 : 1, features.Button)
    if csv_path == dir*"/data/calibration_set/All estimation raw data.csv"
        # Change the RT column to missing if the value is the string "NA", then convert it to a float
        features.RT = map(x -> x == "NA" ? missing : parse(Float64, x), features.RT)
    end
    # Create dummy variables for RT: RT_NA=1 if RT is missing, RT_high=1 if RT is greater than the median
    features[!, :RT_NA] = map(x -> ismissing(x) ? 1 : 0, features.RT)
    # Calculate the median RT value
    median_RT = median(skipmissing(features.RT))
    # Create a new column RT_high that is 1 if RT is greater than median_RT, 0 otherwise
    features[!, :RT_high] = map(x -> ismissing(x) ? 0 : x > median_RT, features.RT)
    # Remove RT column
    features = select(features, Not(:RT))
    # ================================================================================================================= #


    # Generate additional attention/risk features --------------------------------------------------------------------- #
    # Generate an additional attention feature: fatigue index
    features.fatigue = 0.66*features.Order + 0.33*features.Trial
    # Generate an additional attention feature: potential signs of information overload
    features.info_overload = features.LotNumA + features.LotNumB - features.Amb + features.RT_high + features.Feedback
    # Generate an additional risk feature: difference in expected value of high lotteries
    features.diff = features.Hb - features.Ha
    # ================================================================================================================= #


    # Remove attention and risk features if attention_and_risk == 0
    if attention_and_risk == 0
      features = select(features, 
        Not(:LotNumA, :LotNumB, :Corr, :Order, :Trial, :block, :Feedback, r"LotShapeA_.", r"LotShapeB_.", r"RT_.*", 
            :fatigue, :info_overload, :diff))
    end

    # Normalise features and define X, a Matrix of the features
      features = normalise(Matrix(features))
      X = Float32.(Matrix(features))

    # Define Y, the labels
    classes = sort(unique(labels))
    Y = Float32.(Flux.onehotbatch(labels, classes))
    if csv_path == dir*"/data/calibration_set/All estimation raw data.csv"
      data = Iterators.repeated((X, Y), 1000)
    else
      data = (X, Y)
    end

    return X, Y, data
end

# Training data
(X_train, Y_train, train_data) = prepare_data(dir*"/data/calibration_set/All estimation raw data.csv")
# Testing data
(X_test, Y_test, test_data) = prepare_data(dir*"/data/competition_set/raw-comp-set-data-Track-2.csv")

# MODELING ------------------------------------------------------------------------------------------------------------------ #
if attention_and_risk == 0
  model = Chain(
    Dense(16, 16, leakyrelu),
    Dense(16, 16, leakyrelu),
    Dense(16, 12, leakyrelu),
    Dense(12, 10, leakyrelu),
    Dense(10, 2, leakyrelu),
    softmax)
else
  model = Chain(
    Dense(36, 36, leakyrelu),
    Dense(36, 36, leakyrelu),
    Dense(36, 20, leakyrelu),
    Dense(20, 10, leakyrelu),
    Dense(10, 2, leakyrelu),
    softmax)
end

loss(x, y) = Flux.mse(model(x'), y)
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))
optimiser = Descent(0.5)
# ============================================================================================================================ #

Flux.train!(loss, Flux.params(model), train_data, optimiser)

loss(X_train, Y_train)
loss(X_test, Y_test) 
accuracy(X_train', Y_train, model)
accuracy(X_test', Y_test, model)