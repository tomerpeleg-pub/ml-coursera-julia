using LinearAlgebra

function identity(n::Int)::Matrix
    Matrix(1I, n, n)
end

print(identity(5))
