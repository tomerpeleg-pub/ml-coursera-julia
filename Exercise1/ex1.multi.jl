using CSV
using Plots
using Tables
using LinearAlgebra
using Statistics

function getData(path)
    return CSV.File(path, datarow = 1) |> Tables.matrix
end

function featureNormalize(X)
    mu = map(mean, eachcol(X))
    sigma = map(std, eachcol(X))
    X_norm = (X .- transpose(mu)) / transpose(sigma)

    return (X_norm, mu, sigma)
end

function featureDeNormalize(X, mu, sigma)
    return (X * sigma) .+ transpose(mu)
end


function computeCostMulti(X, y, theta)
    m = length(y)
    error = X * theta - y
    return (transpose(error) * error) / (2 * m)
end


function gradientDescent(X, y, theta, alpha, iterations)
    # Initialize some useful values
    m = length(y)

    J_history = []

    for i = 1:iterations

        h = X * theta

        theta = theta - (alpha / m) * transpose(X) * (h - y)
        J = computeCostMulti(X, y, theta)

        push!(J_history, J)
    end

    return (theta, J_history)
end

function ex1Multi()
    gr()
    data = getData("data\\ex1data2.txt")
    m = size(data, 2) - 1
    y = data[:, m+1]
    X = data[:, 1:m]

    plot(data[:, 1], y, seriestype = :scatter, label = "First")
    savefig("output\\ex2.multi.1.png")
    plot(data[:, 2], y, seriestype = :scatter, label = "Second")
    # plot(p1, p2)
    savefig("output\\ex2.multi.2.png")

    X_norm, mu, sigma = featureNormalize(X)

    print("\nGot norms: mu", mu, ", sigma: ", sigma, "X_norm:\n")
    print(X_norm)

    XOne = [ones(size(X, 1)) X_norm]

    theta = zeros(2)          # Initialize fitting parameters
    iterations = 1500
    alpha = 0.03
    theta, J_history = gradientDescent(XOne, y, theta, alpha, iterations)

    print("\nAnswer, theta: ", theta)


    plot(J_history)
    savefig("output\\ex2.J_History.png")

    # XBack = featureDeNormalize(X, mu, sigma) * theta
    XBack = X * theta
    plot(data[:, 1], y, seriestype = :scatter, label = "First")
    plot!(data[:, 1], XBack[:, 1], seriesTypes = :line, label = "Gradient Descent")
    savefig("output\\ex2.multi.1.answer.png")
end

ex1Multi()
