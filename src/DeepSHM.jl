using Flux
using FluxTraining
using CUDA
using HDF5
using MLUtils: splitobs
using Statistics
using FluxTraining.Events: LossBegin, BackwardBegin, BackwardEnd

struct MyTrainingPhase <: FluxTraining.AbstractTrainingPhase end
function FluxTraining.step!(learner, phase::MyTrainingPhase, batch)
    xs, ys = batch |> gpu
    FluxTraining.runstep(learner, phase, (xs=xs, ys=ys)) do handle, state
        state.grads = gradient(learner.params) do
            state.ŷs = learner.model(state.xs)
            handle(LossBegin())
            state.loss = learner.lossfn(state.ŷs, state.ys)
            handle(BackwardBegin())
            return state.loss
        end
        handle(BackwardEnd())
        Flux.update!(learner.optimizer, learner.params, state.grads)
    end
end
struct MyValidationPhase <: FluxTraining.AbstractValidationPhase end
function FluxTraining.step!(learner, phase::MyValidationPhase, batch)
    xs, ys = batch |> gpu
    FluxTraining.runstep(learner, phase, (xs=xs, ys=ys)) do _, state
        state.ŷs = learner.model(state.xs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end

function prepare_data(kmers, freqs, batch_size=32)
    return Flux.DataLoader((kmers, freqs), batchsize=batch_size, shuffle=true, partial=false)
end

function load_data(file_path::String)
    h5open(file_path, "r") do file
        groups = keys(file)
        all_kmers = []
        all_freqs = []
        for group in groups
            kmer = Float32.(Flux.unsqueeze(permutedims(read(file["$group/kmer"]), (3, 2, 1)) .== true, dims=3))
            freq = read(file["$group/freq"])
            push!(all_kmers, kmer)
            push!(all_freqs, freq)
        end        
        return cat(all_kmers..., dims=4), cat(all_freqs..., dims=1)
    end
end

kmers, freq = load_data("data/IGHV_kmer15_by_family.h5")
trainindices, validindices = splitobs(1:size(kmers, 4), at=0.9)
trainiter = prepare_data(kmers[:,:,:,trainindices], freq[trainindices])
validiter = prepare_data(kmers[:,:,:,validindices], freq[validindices])

# Model loaded from Tensroflow models/model_15_mf.h5
# Layer Name: conv2d
# Layer Type: Conv2D
# Input Shape: (None, 4, 15, 1)
# Output Shape: (None, 1, 15, 256)
# Filters: 256
# Kernel Size: (4, 8)
# Strides: (4, 1)
# Padding: same
# Activation: relu
# ---
# Layer Name: dropout
# Layer Type: Dropout
# Input Shape: (None, 1, 15, 256)
# Output Shape: (None, 1, 15, 256)
# Rate: 0.3
# ---
# Layer Name: flatten
# Layer Type: Flatten
# Input Shape: (None, 1, 15, 256)
# Output Shape: (None, 3840)
# ---
# Layer Name: dense
# Layer Type: Dense
# Input Shape: (None, 3840)
# Output Shape: (None, 64)
# Units: 64
# Activation: elu
# ---
# Layer Name: dropout_1
# Layer Type: Dropout
# Input Shape: (None, 64)
# Output Shape: (None, 64)
# Rate: 0
# ---
# Layer Name: dense_1
# Layer Type: Dense
# Input Shape: (None, 64)
# Output Shape: (None, 16)
# Units: 16
# Activation: elu
# ---
# Layer Name: dropout_2
# Layer Type: Dropout
# Input Shape: (None, 16)
# Output Shape: (None, 16)
# Rate: 0.4
# ---
# Layer Name: dense_2
# Layer Type: Dense
# Input Shape: (None, 16)
# Output Shape: (None, 1)
# Units: 1
# Activation: linear
model = Chain(
    Conv((8, 4), 1 => 256, relu; stride=(1, 4), pad=SamePad()),
    Dropout(0.3),
    Flux.flatten,
    Dense(15 * 256, 64, elu),
    # Dropout(0.0), # No dropout
    Dense(64, 16, elu),
    Dropout(0.4),
    Dense(16, 1, identity),
    x->dropdims(x, dims=1)
) |> gpu

backend = TensorBoardBackend("logs/deepshm")

learner = Learner(model, Flux.Losses.mse,
    data = (trainiter, validiter),
    optimizer=Flux.RMSProp(1e-4),
    callbacks=[Metrics(Metric(Flux.Losses.mae, name="Mean Absolute Error")), LogMetrics(backend), Checkpointer("models/DeepSHM.jld2", keep_top_k=1)],
)

# Train
for epoch in 1:4000
    FluxTraining.epoch!(learner, MyTrainingPhase(), trainiter)
    FluxTraining.epoch!(learner, MyValidationPhase(), validiter)
end