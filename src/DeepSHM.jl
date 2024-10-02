using Flux
using FluxTraining
using CUDA
using HDF5
using MLUtils: splitobs
using Statistics
using FluxTraining.Events: LossBegin, BackwardBegin, BackwardEnd
using JLD2
using FASTX
using OneHotArrays
using CSV
using DataFrames

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

function save_model(model, epoch, filename="models/DeepSHM.jld2")
    model_state = Flux.state(cpu(model))
    jldsave(filename; model_state, epoch)
    println("Model saved to $filename")
end

function load_model!(model, filename="models/DeepSHM.jld2")
    data = load(filename)
    Flux.loadmodel!(model, data["model_state"])
    println("Model loaded from $filename")
    return data["epoch"]
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
epochs = 4000
learner = Learner(model, Flux.Losses.mse,
    data = (trainiter, validiter),
    optimizer=Flux.RMSProp(1e-4),
    callbacks=[Metrics(Metric(Flux.Losses.mae, name="Mean Absolute Error")), LogMetrics(backend)],
)
# Train
# for epoch in 1:epochs
#     FluxTraining.epoch!(learner, MyTrainingPhase(), trainiter)
#     FluxTraining.epoch!(learner, MyValidationPhase(), validiter)
# end

# save_model(model, epochs, "models/DeepSHM.jld2")
load_model!(model, "models/DeepSHM.jld2")

#
# Do predictions for sequences in FASTA file
#

function load_fasta(fasta_file::String)
    sequences = Vector{Tuple{String,String}}()
    open(fasta_file, "r") do io
        reader = FASTA.Reader(io)
        for record in reader
            seq_id = FASTA.identifier(record)
            sequence = FASTA.sequence(String, record)
            push!(sequences, (seq_id, sequence))
        end
    end
    # Sort the sequences by name (first element of each tuple)
    sort!(sequences, by = x -> x[1])
    return sequences
end

function generate_kmers_for_sequences(sequences::Vector{Tuple{String,String}}, k::Int)
    kmer_sequences = Vector{Tuple{String,Vector{String}}}()
    for (seq_id, sequence) in sequences
        kmers = String[]
        for i in 1:(length(sequence) - k + 1)
            push!(kmers, sequence[i:i+k-1])
        end
        push!(kmer_sequences, (seq_id, kmers))
    end
    return kmer_sequences
end

function encode_kmers(kmer_sequences::Vector{Tuple{String,Vector{String}}})
    encoded_sequences = Vector{Tuple{String,Array{Bool,3}}}()
    nucleotides = ['A', 'C', 'G', 'T']
    for (seq_id, kmers) in kmer_sequences
        k = length(first(kmers))
        encoded_kmers = falses(4, k, length(kmers))
        for (i, kmer) in enumerate(kmers)
            for (j, nucleotide) in enumerate(kmer)
                if nucleotide in nucleotides
                    encoded_kmers[findfirst(==(nucleotide), nucleotides), j, i] = true
                end
            end
        end
        push!(encoded_sequences, (seq_id, encoded_kmers))
    end
    return encoded_sequences
end

@info "Predicting mutation frequency for sequences in FASTA file"
loaded_sequences = load_fasta("data/Macaca_mulatta_V.fasta")
kmer_sequences = generate_kmers_for_sequences(loaded_sequences, 15)
encoded_sequences = encode_kmers(kmer_sequences)
data = last.(encoded_sequences)
# Reshape for the model
x = permutedims(Flux.unsqueeze(cat(map(x->x[:,:,:], data)...,dims=3), dims=3),(2,1,3,4)) |> gpu
x̂ = cpu(model(x))
output = DataFrame(kmer=vcat(last.(kmer_sequences)...), mutation_frequency=x̂)
@info "Preductions saved in predictions.tsv"
CSV.write("predictions.tsv", output, delim='\t')
