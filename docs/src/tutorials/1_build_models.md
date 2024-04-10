# How to build a microstructure model

## diffusion MRI model

Load the module

````julia
using Microstructure
````

Specify the acquisition parameters and make a protocol. In real case, you can read a protocol from your acquisition text files

````julia
bval = [1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000].*1.0e6
techo = 40.0.*ones(8,).*1e-3
tdelta = 15.192.*ones(8,).*1e-3
tsmalldel = 11.0.*ones(8,).*1e-3
prot = Protocol(bval,techo,tdelta,tsmalldel)
````

Specify a model containing all the tissue parameters. Here, the example ExCaliber is a model for estimating axon diameter in ex vivo tissue using the spherical mean technique

````julia
estimates = ExCaliber()
````

You can check how the tissue is modelled by printing the model. It will give you all the tissue compartments

````julia
print_model(estimates)
````

You can check the values in the tissue model by using @show macro.
This will show the default values if you didn't specify parameters when declare a model

````julia
@show estimates
````

You can specify tissue parameters when declearing a model; fields/subfiedls that are not specified will take the default values

````julia
estimates = ExCaliber( axon = Cylinder(da = 4e-6, dpara = 0.7e-9))
````

You can change the fields/subfields of a decleared model struct by using update! funciton

update a field/subfields

````julia
undate!(estimates, "axon.da" => 5e-6)
````

update a field/subfield using parameter links.
It's common that we need to link certain tissue parameters in some models as they may not be distinguishable under the experimental condition

````julia
update!(estimates,"axon.d0" => "axon.dpara")
````

update multiple parameters

````julia
update!(estimates,("axon.da" => 5e-6, "axon.dpara" => 0.5e-9, "axon.d0" => "axon.dpara", "extra.dpara" => "axon.dpara"))
````

Now we can use the model and protocol to generate some mri signals

````julia
signals = model_signals(estimates,prot)
````

We can add some noise to the signals to make them look like real measurements

````julia
using Random, Distributions

sigma = 0.01 # SNR=100 at S(b=0,t=TEmin) (b=0 of minimal TE)
noise = Normal(0,sigma)
````

Add some Gaussian noise

````julia
meas = signals .+ rand(noise,size(signals))
````

or Rician noise

````julia
meas_rician = sqrt.((signals .+ rand(noise,size(signals))).^2.0 .+ rand(noise,size(signals)).^2.0)
````

You can check the predict signals and simulated measurements by ploting them

````julia
using Plots
plot(prot.bval, signals, label="predicted signals", lc=:black, lw=2)
scatter!(prot.bval, meas, label="noisy measurements", mc=:red, ms=2, ma=0.5)
xlabel!("b-values [s/m^2]")
````
## Combined Diffusion-relaxometry model 

Now let's look at a diffusion-relaxometry model MTE-SANDI. Similarly, declear a model object and check the values

````julia
model = MTE_SANDI()
print_model(model)
@show model
````

MTE_SANDI requires data acquired at multiple echo times to solve the inverse problem and we will define a different protocol for it make a multi-TE protocol

````julia
nTE = 4
nb = 9
bval = repeat([0, 1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000].*1.0e6, outer=nTE)
techo = repeat([32, 45, 60, 78].*1e-3, inner=9)
tdelta = 15.192.*ones(nTE*nb,).*1e-3
tsmalldel = 11.0.*ones(nTE*nb,).*1e-3
prot = Protocol(bval,techo,tdelta,tsmalldel)
````

Let's see how multi-TE signals look like

````julia
signals = model_signals(model, prot)
meas = signals .+ rand(noise,size(signals))

plot(signals, label="predicted signals", lc=:black, lw=2)
scatter!(meas, label="noisy measurements", mc=:red, ms=2, ma=0.5)
````



