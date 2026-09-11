using DataFrames
using Plots
using CSV

default(dpi=300)

eilmer = CSV.read("eilmer.dat", delim=' ', ignorerepeated=true, DataFrame)
nesl = CSV.read("klomfass.dat", delim=',', stripwhitespace=true, DataFrame)

x_eilmer = eilmer."pos.x" .- eilmer."pos.x"[end] .+ nesl."x (m)"[end]
x_nesl = nesl."x (m)"

plot(x_eilmer, eilmer.T, label="Eilmer", xlabel="x, m", ylabel="Temperature, K", linewidth=2, framestyle = :box)
plot!(x_nesl, nesl."T (K)", label="NESL", linewidth=2)
plot!(legend=:outertopright)
savefig("temp.png")

plot(x_nesl, nesl.N2, label="NESL N2", xlabel="x, m", ylabel="Mass fraction", linewidth=2, framestyle = :box)
plot!(x_eilmer, eilmer."massf-N2", label="Eilmer N2", linestyle=:dash, linewidth=2)
plot!(x_nesl, nesl.O2, label="NESL O2", linewidth=2)
plot!(x_eilmer, eilmer."massf-O2", label="Eilmer O2", linestyle=:dash, linewidth=2)
plot!(x_nesl, nesl.NO, label="NESL NO", linewidth=2)
plot!(x_eilmer, eilmer."massf-NO", label="Eilmer NO", linestyle=:dash, linewidth=2)
plot!(x_nesl, nesl.N, label="NESL N", linewidth=2)
plot!(x_eilmer, eilmer."massf-N", label="Eilmer N", linestyle=:dash, linewidth=2)
plot!(x_nesl, nesl.O, label="NESL O", linewidth=2)
plot!(x_eilmer, eilmer."massf-O", label="Eilmer O", linestyle=:dash, linewidth=2)
plot!(legend=:outertopright)
savefig("massf.png")
