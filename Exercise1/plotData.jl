using CSV
using Plots
using Tables
using LinearAlgebra

function getData(path)
    return CSV.File(path, datarow = 1) |> Tables.matrix
end

function plotChart(X, y, outputPath)
    scatter(
        X,
        y,
        label = "House prices",
        xlabel = "Popluation of City in 10,000s",
        ylabel = "Profit in \$10,000s",
    )
    savefig(outputPath)
end

function computeCost(X, y, theta)
    m = length(y)
    error = X * theta - y
    return dot(error, error) / (2 * m)
end

function gradientDescent(X, y, theta, alpha, iterations)
    # Initialize some useful values
    m = length(y)

    J_history = []

    for i = 1:iterations

        h = X * theta

        theta = theta - (alpha / m) * transpose(X) * (h - y)
        J = computeCost(X, y, theta)

        push!(J_history, J)
    end

    return (theta, J_history)
end

function ex1()
    gr()
    data = getData("data\\ex1data1.txt")

    X = data[:, 1] # Population
    y = data[:, 2] # Profit
    m = length(y)  # Number of samples

    # 2.1 Plotting data
    # plotChart(X, y, "output/plotData.png")

    # 2.2 Gradient Descent
    X = [ones(m, 1) data[:, 1]] # Add a column of 1's to X
    theta = zeros(2)          # Initialize fitting parameters

    iterations = 1500
    alpha = 0.01

    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

    plot(
        data[:, 1],
        y,
        seriestype = :scatter,
        label = "House prices",
        xlabel = "Popluation of City in 10,000s",
        ylabel = "Profit in \$10,000s",
    )
    plot!(data[:, 1], X * theta, seriesTypes = :line, label = "Gradient Descent")
    # plot(p1, p2)
    savefig("output\\gradientDescent.png")

    plot(J_history)
    savefig("output\\J_History.png")

    print("Final:", theta)
end

ex1()
