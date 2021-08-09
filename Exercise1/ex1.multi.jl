using CSV
using Plots
using Tables
using LinearAlgebra
using Statistics

function getData(path)
    return CSV.File(path, datarow = 1) |> Tables.matrix
end

function featureNormalize(X)

    mu = mapslices(mean, X, dims = 1)

    XM = X .- mu
    sigma = mapslices(std, XM, dims = 1)
    X_norm = XM ./ sigma

    return (X_norm, mu, sigma)
end

function featureDeNormalize(X, mu, sigma)
    return (X .* sigma) .+ mu
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


    XOne = [ones(size(X, 1)) X_norm]

    theta = zeros(3)          # Initialize fitting parameters
    iterations = 1500
    alpha = 0.1
    theta, J_history = gradientDescent(XOne, y, theta, alpha, iterations)



    plot(J_history)
    savefig("output\\ex2.J_History.png")

    # XBack = featureDeNormalize(X, mu, sigma) * theta
    Test = [1 -2 -2; 1 0 0; 1 2 2]
    TestY = Test * theta
    TestUnNormalized = featureDeNormalize(Test[:, 2:3], mu, sigma)

    print("\nFinal theta: ", theta, ", example: ", TestUnNormalized, "\n\n")

    plot(data[:, 1], y, seriestype = :scatter, label = "First")
    plot!(TestUnNormalized[:, 1], TestY, seriesTypes = :line, label = "Gradient Descent")
    savefig("output\\ex2.multi.1.answer.png")
    plot(data[:, 2], y, seriestype = :scatter, label = "Second")
    plot!(TestUnNormalized[:, 2], TestY, seriesTypes = :line, label = "Gradient Descent")
    savefig("output\\ex2.multi.2.answer.png")
end

ex1Multi()
