using JLD2
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using CUDSS

A_cpu=load("matrix.jld2","A")
T=eltype(A)
n=size(A,1)
x_cpu = zeros(T, n)
b_cpu = rand(T, n)

A_gpu = CuSparseMatrixCSR(A_cpu)
x_gpu = CuVector(x_cpu)
b_gpu = CuVector(b_cpu)

solver = CudssSolver(A_gpu, "G", 'F')
cudss_set(solver,"ir_tol",1E-14)
cudss("analysis", solver, x_gpu, b_gpu)
cudss("factorization", solver, x_gpu, b_gpu)
cudss("solve", solver, x_gpu, b_gpu)

r_gpu = b_gpu - A_gpu * x_gpu
norm(r_gpu)

F=lu(A_cpu)
x_cpu = F \ b_cpu
r_cpu = b_cpu - A_cpu * x_cpu
norm(r_cpu)

F_gpu=lu(A_gpu)
x_gpu = F_gpu \ b_gpu
r_gpu = b_gpu - A_gpu * x_gpu
norm(r_gpu)