using Flux, DifferentialEquations, ModelingToolkit
using DiffEqBase, SciMLBase
using GalacticOptim, Optim
using Quadrature
using DomainSets
using NeuralPDE

@parameters x,y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
Ix = Integral((x,y) in DomainSets.UnitSquare())
eq = Ix(u(x,y)) ~ 1/3
bcs = [u(0., 0.) ~ 1, Dx(u(x,y)) ~ -2*x , Dy(u(x ,y)) ~ -2*y ]
domains = [x ∈ Interval(0.0,1.00), y ∈ Interval(0.0,1.00)]
chain = Chain(Dense(2,15,Flux.σ),Dense(15,1))
initθ = Float64.(DifferentialEquations.initial_params(chain))
strategy_ = NeuralPDE.GridTraining(0.1)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy_;
                                             init_params = nothing,
                                             phi = nothing,
                                             derivative = nothing,
                                             )
@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = NeuralPDE.discretization(pde_system,discretization)
cb = function (p,l)
    println("Current loss is: $l")
    return false
end
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=100)
xs = 0.00:0.01:1.00
ys = 0.00:0.01:1.00
phi = discretization.phi

u_real = collect(1 - x^2 - y^2 for y in ys, x in xs);
u_predict = collect(Array(phi([x,y], res.minimizer))[1] for y in ys, x in xs);

using Plots
error_ = u_predict .- u_real
p1 = Plot(xs,ys,u_real,linetype=:contourf,label = "analytic")
p2 = Plot(xs,ys,u_predict,linetype=:contourf,label = "predict")
p3 = Plot(xs,ys,error_,linetype=:contourf,label = "error")
plot(p1,p2,p3)

https://nextjournal.com/ashutosh-b-b/gsoc-3