using Flux
using Statistics
using Random
using ProgressMeter
using HDF5
using MLUtils: splitobs
using CUDA
using JLD2
using CairoMakie
using CSV
using DataFrames

const SEED = 42

function generate_model(
    input_shape,
    channels_conv,
    kernel_sizes,
    pooling,
    dropout_conv,
    channels_fc,
    dropout_fc,
    task;
    padding=SamePad(),
    initializer=Flux.glorot_normal,
    activation=Flux.elu
)
    Random.seed!(SEED)
    layers = []

    # Convolutional layers
    for i in eachindex(channels_conv)
        if i == 1
            in_channels = input_shape[3]  # Assuming input_shape is (width, height, channels, batch_size)
            push!(layers, Conv(kernel_sizes[i], in_channels => channels_conv[i], activation, pad=padding, init=initializer))
        else
            push!(layers, Conv(kernel_sizes[i], channels_conv[i-1] => channels_conv[i], activation, pad=padding, init=initializer))
        end

        if pooling[i] > 1
            push!(layers, MeanPool((pooling[i], 1)))  # Pooling along sequence length
        end

        push!(layers, Dropout(dropout_conv[i]))
    end

    push!(layers, Flux.flatten)

    # Calculate the size after flattening
    conv_output_width = input_shape[1]  # sequence length
    conv_output_height = input_shape[2]  # nucleotide dimension
    for i in eachindex(channels_conv)
        if pooling[i] > 1
            conv_output_width = conv_output_width ÷ pooling[i]
        end
    end
    conv_output_size = conv_output_width * conv_output_height * channels_conv[end]

    # Fully connected layers
    for i in eachindex(channels_fc)
        push!(layers, Dense(i == 1 ? conv_output_size : channels_fc[i-1], channels_fc[i], activation, init=initializer))
        i < length(channels_fc) && push!(layers, Dropout(dropout_fc[i]))
    end

    # Output layer
    if task == "mut_freq"
        push!(layers, Dense(channels_fc[end], 1, init=initializer))
        push!(layers, x->dropdims(x, dims=1))
    elseif task in ["weighted_sub", "substitution"]
        push!(layers, Dense(channels_fc[end], 4, init=initializer))
    end

    return Chain(layers...)
end

channels_conv = [32, 64, 128]
kernel_sizes = [(3, 4), (3, 1), (3, 1)]
pooling = [2, 2, 0]
dropout_conv = [0.1, 0.1, 0.1]
channels_fc = [128, 64]
dropout_fc = [0.5, 0.5]
task = "mut_freq"

model = generate_model((15, 4, 1, 32), channels_conv, kernel_sizes, pooling, dropout_conv, channels_fc, dropout_fc, task) |> gpu

function prepare_data(kmers, freqs, batch_size=32)
    return Flux.DataLoader((kmers, freqs), batchsize=batch_size, shuffle=true, partial=false)
end

function process_h5_file(file_path::String)
    h5open(file_path, "r") do file
        groups = keys(file)
        all_kmers = []
        all_freqs = []
        for group in groups
            kmer = Flux.unsqueeze(permutedims(read(file["$group/kmer"]), (3,2,1)), dims=3)
            freq = read(file["$group/freq"])
            push!(all_kmers, kmer)
            push!(all_freqs, freq)
        end        
        return cat(all_kmers..., dims=4), cat(all_freqs..., dims=1)
    end
end

kmer_array, freq_vector = process_h5_file("data/IGHV_kmer15_by_family.h5")
trainindices, testindices = splitobs(1:size(kmer_array, 4), at=0.9)
trainset = prepare_data(kmer_array[:,:,:,trainindices], freq_vector[trainindices])
testset = prepare_data(kmer_array[:,:,:,testindices], freq_vector[testindices])

function train!(model, train_data, test_data; 
                n_epochs=50, 
                learning_rate=0.0001, 
                optimizer=Flux.RMSProp, 
                loss=Flux.mse, 
                save_path=nothing,
                eval_frequency=10,
                use_gpu=true,
                log_file="training_log.tsv")
    opt = optimizer(learning_rate)
    opt_state = Flux.setup(opt, model)

    # Initialize log file with headers
    open(log_file, "w") do io
        println(io, "Epoch\tTraining Loss\tTraining MAE\tTest Loss\tTest MAE")
    end

    for epoch in 1:n_epochs
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        for (x, y) in (use_gpu ? CUDA.CuIterator(train_data) : train_data)
            loss_val, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                loss(ŷ, y)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += loss_val
            total_mae += mean(abs.(model(x) .- y))
            num_batches += 1
        end
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        if epoch % eval_frequency == 0
            println("Epoch $epoch: Average Training Loss = $avg_loss, MAE = $avg_mae")

            # Evaluation step
            test_loss = 0.0
            test_mae = 0.0
            test_batches = 0
            for (x, y) in (use_gpu ? CUDA.CuIterator(test_data) : test_data)
                ŷ = model(x)
                test_loss += loss(ŷ, y)
                test_mae += mean(abs.(ŷ .- y))
                test_batches += 1
            end
            avg_test_loss = test_loss / test_batches
            avg_test_mae = test_mae / test_batches
            println("Epoch $epoch: Average Test Loss = $avg_test_loss, MAE = $avg_test_mae")

            # Log losses and MAE to file
            open(log_file, "a") do io
                println(io, "$epoch\t$avg_loss\t$avg_mae\t$avg_test_loss\t$avg_test_mae")
            end
        end
    end

    if !isnothing(save_path)
        model_state = Flux.state(cpu(model))
        jldsave(save_path; model_state)
    end
end

train!(model, trainset, testset, n_epochs=2000, save_path="results/DeepSHM-2000epochs.jld2", log_file="results/training_log.tsv")

function plot_learning_curves(log_file, output_path="results/learning_curves.png")
    df = CSV.read(log_file, DataFrame, delim='\t')

    fig = Figure(size=(1000, 800))

    # Loss plot
    ax1 = Axis(fig[1, 1],
               xlabel = "Epoch",
               ylabel = "Loss",
               title = "Learning Curves - Loss")

    lines!(ax1, df.Epoch, df."Training Loss", label="Training")
    lines!(ax1, df.Epoch, df."Test Loss", label="Test")

    axislegend(ax1)

    # MAE plot
    ax2 = Axis(fig[2, 1],
               xlabel = "Epoch",
               ylabel = "MAE",
               title = "Learning Curves - MAE")

    lines!(ax2, df.Epoch, df."Training MAE", label="Training")
    lines!(ax2, df.Epoch, df."Test MAE", label="Test")

    axislegend(ax2)

    save(output_path, fig)
end

plot_learning_curves("results/training_log.tsv", "results/learning_curves.png")