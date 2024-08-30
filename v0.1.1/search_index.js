var documenterSearchIndex = {"docs":
[{"location":"manual/models/#Microstructure-Models","page":"Microstructure Models","title":"Microstructure Models","text":"","category":"section"},{"location":"manual/models/#dMRI-models","page":"Microstructure Models","title":"dMRI models","text":"","category":"section"},{"location":"manual/models/#WM-models","page":"Microstructure Models","title":"WM models","text":"","category":"section"},{"location":"manual/models/","page":"Microstructure Models","title":"Microstructure Models","text":"ExCaliber","category":"page"},{"location":"manual/models/#Microstructure.ExCaliber","page":"Microstructure Models","title":"Microstructure.ExCaliber","text":"ExCaliber(axon, extra, csf, dot, fracs)\n\nExCaliber model for ex vivo tissue; dot signal considered The fraction vector represents fractions of the axon, CSF and dot with the fraction of extra being 1-sum(fracs)\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#GM-models","page":"Microstructure Models","title":"GM models","text":"","category":"section"},{"location":"manual/models/","page":"Microstructure Models","title":"Microstructure Models","text":"SANDI","category":"page"},{"location":"manual/models/#Microstructure.SANDI","page":"Microstructure Models","title":"Microstructure.SANDI","text":"SANDI(soma, neurite, extra, fracs)\n\nSANDI models or MTE-SANDI models including three tissue compartments for in vivo imaging For SANDI model, ignore the field of t2 in all compartments and set them to 0\n\n\n\n\n\n","category":"type"},{"location":"manual/models/","page":"Microstructure Models","title":"Microstructure Models","text":"SANDIdot","category":"page"},{"location":"manual/models/#Microstructure.SANDIdot","page":"Microstructure Models","title":"Microstructure.SANDIdot","text":"SANDIdot models or MTE-SANDIdot models including additionally a dot compartment for ex vivo imaging For SANDIdot model, ignore the field of t2 in all compartments and set them to 0 For MTE-SANDIdot model, consider the t2 values in all compartments The fraction vector represents fractions of the soma, neurite and dot with the fraction of extra being 1-sum(fracs)\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#Combined-diffusion-relaxometry-models","page":"Microstructure Models","title":"Combined diffusion-relaxometry models","text":"","category":"section"},{"location":"manual/models/","page":"Microstructure Models","title":"Microstructure Models","text":"MTE_SMT","category":"page"},{"location":"manual/models/#Microstructure.MTE_SMT","page":"Microstructure Models","title":"Microstructure.MTE_SMT","text":"MTE_SMT(axon, extra, fracs, S0norm)\n\nTo test multi-TE spherical mean technique for low-b in vivo imaging\n\n\n\n\n\n","category":"type"},{"location":"manual/models/","page":"Microstructure Models","title":"Microstructure Models","text":"MTE_SANDI","category":"page"},{"location":"manual/models/#Microstructure.MTE_SANDI","page":"Microstructure Models","title":"Microstructure.MTE_SANDI","text":"MTE_SANDI(soma, neurite, extra, fracs, S0norm)\n\nFor MTE-SANDI model, consider the t2 values in all compartments The fraction vector represents fractions of the soma and neurite with the fraction of extra being 1-sum(fracs) S0norm is the relaxation-weighting free signal from all compartments S(b=0,t=0) normalised by S(b=0,t=TEmin)\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#Prediction-of-MRI-signals","page":"Microstructure Models","title":"Prediction of MRI signals","text":"","category":"section"},{"location":"manual/models/","page":"Microstructure Models","title":"Microstructure Models","text":"model_signals","category":"page"},{"location":"manual/models/#Microstructure.model_signals","page":"Microstructure Models","title":"Microstructure.model_signals","text":"model_signals(model,prot[,links])\n\nPredict model signals from BiophysicalModel model and imaging protocol 'prot'.     links is a optional argument that specify parameter links in the model\n\n\n\n\n\n","category":"function"},{"location":"tutorials/1_build_models/#How-to-build-a-microstructure-model","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"","category":"section"},{"location":"tutorials/1_build_models/#1.-diffusion-MRI-model","page":"How to build a microstructure model","title":"1. diffusion MRI model","text":"","category":"section"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Load the module","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"using Microstructure","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Specify the acquisition parameters and make a protocol. In real case, you can read a protocol from your acquisition text files","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"bval = [1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000].*1.0e6\ntecho = 40.0.*ones(8,).*1e-3\ntdelta = 15.192.*ones(8,).*1e-3\ntsmalldel = 11.0.*ones(8,).*1e-3\nprot = Protocol(bval,techo,tdelta,tsmalldel)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Specify a model containing all the tissue parameters. Here, the example ExCaliber is a model for estimating axon diameter in ex vivo tissue using the spherical mean technique","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"estimates = ExCaliber()","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"You can check how the tissue is modelled by printing the model. It will give you all the tissue compartments","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"print_model(estimates)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"You can check the values in the tissue model by using @show macro. This will show the default values if you didn't specify parameters when declare a model","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"@show estimates","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"You can specify tissue parameters when declearing a model; fields/subfiedls that are not specified will take the default values","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"estimates = ExCaliber( axon = Cylinder(da = 4e-6, dpara = 0.7e-9))","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"You can change the fields/subfields of a decleared model struct by using update! funciton.","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"a. update a field/subfields","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"undate!(estimates, \"axon.da\" => 5e-6)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"It's common that we need to link certain tissue parameters in some models as they may not be distinguishable under the experimental condition.","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"b. update a field/subfield using parameter links.","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"update!(estimates,\"axon.d0\" => \"axon.dpara\")","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"c. update multiple parameters","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"update!(estimates,(\"axon.da\" => 5e-6, \"axon.dpara\" => 0.5e-9, \"axon.d0\" => \"axon.dpara\", \"extra.dpara\" => \"axon.dpara\"))","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Now we can use the model and protocol to generate some mri signals","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"signals = model_signals(estimates,prot)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"We can add some noise to the signals to make them look like real measurements","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"using Random, Distributions\n\nsigma = 0.01 # SNR=100 at S(b=0,t=TEmin) (b=0 of minimal TE)\nnoise = Normal(0,sigma)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Add some Gaussian noise","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"meas = signals .+ rand(noise,size(signals))","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"or Rician noise","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"meas_rician = sqrt.((signals .+ rand(noise,size(signals))).^2.0 .+ rand(noise,size(signals)).^2.0)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"You can check the predict signals and simulated measurements by ploting them","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"using Plots\nplot(prot.bval, signals, label=\"predicted signals\", lc=:black, lw=2)\nscatter!(prot.bval, meas, label=\"noisy measurements\", mc=:red, ms=2, ma=0.5)\nxlabel!(\"b-values [s/m^2]\")","category":"page"},{"location":"tutorials/1_build_models/#2.-Combined-Diffusion-relaxometry-model","page":"How to build a microstructure model","title":"2. Combined Diffusion-relaxometry model","text":"","category":"section"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Now let's look at a diffusion-relaxometry model MTE-SANDI. Similarly, declear a model object and check the values","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"model = MTE_SANDI()\nprint_model(model)\n@show model","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"MTE_SANDI requires data acquired at multiple echo times to solve the inverse problem and we will define a different protocol for it.","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Make a multi-TE protocol","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"nTE = 4\nnb = 9\nbval = repeat([0, 1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000].*1.0e6, outer=nTE)\ntecho = repeat([32, 45, 60, 78].*1e-3, inner=9)\ntdelta = 15.192.*ones(nTE*nb,).*1e-3\ntsmalldel = 11.0.*ones(nTE*nb,).*1e-3\nprot = Protocol(bval,techo,tdelta,tsmalldel)","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"Let's see how multi-TE signals look like","category":"page"},{"location":"tutorials/1_build_models/","page":"How to build a microstructure model","title":"How to build a microstructure model","text":"signals = model_signals(model, prot)\nmeas = signals .+ rand(noise,size(signals))\n\nplot(signals, label=\"predicted signals\", lc=:black, lw=2)\nscatter!(meas, label=\"noisy measurements\", mc=:red, ms=2, ma=0.5)","category":"page"},{"location":"tutorials/5_model_selection/#Which-model-to-use","page":"Which model to use","title":"Which model to use","text":"","category":"section"},{"location":"tutorials/5_model_selection/","page":"Which model to use","title":"Which model to use","text":"Constructing...","category":"page"},{"location":"manual/dMRI/#I/O-functions","page":"I/O functions","title":"I/O functions","text":"","category":"section"},{"location":"manual/dMRI/#dMRI-and-Protocol-type","page":"I/O functions","title":"dMRI and Protocol type","text":"","category":"section"},{"location":"manual/dMRI/","page":"I/O functions","title":"I/O functions","text":"dMRI","category":"page"},{"location":"manual/dMRI/#Microstructure.dMRI","page":"I/O functions","title":"Microstructure.dMRI","text":"dMRI(nifti,tdelta,dsmalldel,techo,smt)\n\ncontain additional volume-wise experimental settings that are needed for advanced modelling expects addtional text files saved besides .bvals/.bvecs expected file extensions: .techo/.tdelta/.tsmalldel\n\n\n\n\n\n","category":"type"},{"location":"manual/dMRI/","page":"I/O functions","title":"I/O functions","text":"Protocol","category":"page"},{"location":"manual/dMRI/#Microstructure.Protocol","page":"I/O functions","title":"Microstructure.Protocol","text":"Struct to hold parameters in acquisition protocol relavent for modelling Suggest to use SVector when using smt-based modelling and Vector when using  orientation models where the number of measurements exceeds 50\n\nunit convention: most text files use s/mm^2 for b-values and ms for time\nwe use US unit for microsturcture computation\nb-values (s/m^2); time (s); size (m); G (T/m) (660 mT/m => 0.66T/m)\n\nConsider to add bvec for orientation modelling     or create a new protocol type with gradient orientation info\n\n\n\n\n\n","category":"type"},{"location":"manual/dMRI/#Read-dMRI-data-and-get-spherical-mean-signals-and-imaging-protocols","page":"I/O functions","title":"Read dMRI data and get spherical mean signals & imaging protocols","text":"","category":"section"},{"location":"manual/dMRI/","page":"I/O functions","title":"I/O functions","text":"Spherical_mean","category":"page"},{"location":"manual/dMRI/#write-dMRI-and-save-protocol","page":"I/O functions","title":"write dMRI and save protocol","text":"","category":"section"},{"location":"manual/dMRI/","page":"I/O functions","title":"I/O functions","text":"dmri_write","category":"page"},{"location":"manual/dMRI/#Microstructure.dmri_write","page":"I/O functions","title":"Microstructure.dmri_write","text":"write dmri volume and protocols/b-tables\n\n\n\n\n\n","category":"function"},{"location":"tutorials/2_quality_of_fit/#How-to-check-quality-of-fitting-and-mcmc-samples","page":"How to check quality of fitting and mcmc samples","title":"How to check quality of fitting and mcmc samples","text":"","category":"section"},{"location":"tutorials/2_quality_of_fit/","page":"How to check quality of fitting and mcmc samples","title":"How to check quality of fitting and mcmc samples","text":"Constructing...","category":"page"},{"location":"guide/#Developer-guide","page":"Developer guide","title":"Developer guide","text":"","category":"section"},{"location":"manual/estimators/#Estimators","page":"Estimators","title":"Estimators","text":"","category":"section"},{"location":"manual/estimators/#MCMC","page":"Estimators","title":"MCMC","text":"","category":"section"},{"location":"manual/estimators/#1.-Define-a-sampler-for-your-model","page":"Estimators","title":"1. Define a sampler for your model","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"Sampler","category":"page"},{"location":"manual/estimators/#Microstructure.Sampler","page":"Estimators","title":"Microstructure.Sampler","text":"Sampler for a biophysical model     params: parameters (String) to sample in the model     priorrange: bounds for each parameter     proposal: Distribution to draw pertubations     paralinks: linking parameters in the model     nsamples: The total number of samples in a MCMC chain; default to 70000     burnin: The number of samples that will be discarded in the beginning of the chain; default to 20000     thinning: The interval to extract unrelated samples in the chain; default to 100 Example Sampler for ExCaliber     Sampler(         params = (\"axon.da\",\"axon.dpara\",\"extra.dperpfrac\",\"fracs\")         prior_range = ((1.0e-7,1.0e-5),(0.01e-9,0.9e-9),(0.0, 1.0),(0.0,1.0))         proposal = (Normal(0,0.25e-6), Normal(0,0.025e-9), Normal(0,0.05), MvNormal([0.0025 0 0;0 0.0001 0; 0 0 0.0001])) #; (0,0.05),Normal(0,0.01),Normal(0,0.01)]]         paralinks = (\"axon.d0\" => \"axon.dpara\", \"extra.dpara\" => \"axon.dpara\")     )\n\n\n\n\n\n","category":"type"},{"location":"manual/estimators/#2.-Define-a-noise-model","page":"Estimators","title":"2. Define a noise model","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"Noise_model","category":"page"},{"location":"manual/estimators/#3.-Run-MCMC-on-your-model-and-data","page":"Estimators","title":"3. Run MCMC on your model and data","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"mcmc!","category":"page"},{"location":"manual/estimators/#Microstructure.mcmc!","page":"Estimators","title":"Microstructure.mcmc!","text":"Run mcmc for a model and sampler\n\nMethod 1 generates pertubations within function, creates and returns a dict chain, and modify final model estimates in place.     This method is useful in checking a few voxels, e.g. for quality of fitting, chain dignostics and optimizing sampler for models. \n\nMethod 2 takes chain and pertubations as input, mutating chain in place which can be used to calculate finial estimates and uncertainties.      This method is used for processing larger dataset, e.g. for whole-barin/slices.      This method is used together with multi-threads processing that pre-allocate spaces for caching chains, avoiding creating them for each voxel.      This method also reuses pertubations for faster speed, as we usually use a very large number of pertubations (e.g. 70000) to densely sample the proposal distributions. \n\nMethod 1 \n\njulia> mcmc!(estimates,measurements,protocol,sampler,noise_model,rng)\n\nMethod 2: 'chain' can be Vector (modify elements) or Dict (push!); need to benchmark time difference\n\njulia> mcmc!(chain,estimates,meas,protocol,sampler,pertubations,noise_model))\n\n\n\n\n\n","category":"function"},{"location":"manual/estimators/#Neural-Networks","page":"Estimators","title":"Neural Networks","text":"","category":"section"},{"location":"manual/estimators/#1.-Specify-a-network-model-for-your-task","page":"Estimators","title":"1. Specify a network model for your task","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"NetworkArg","category":"page"},{"location":"manual/estimators/#Microstructure.NetworkArg","page":"Estimators","title":"Microstructure.NetworkArg","text":"NetworkArg(model, protocol, params, paralinks, tissuetype, sigma, noise_type, \nhidden_layers, nsamples, nin, nout, dropoutp)\n\ncontain fields that determine network architecture and training samples for a biophysical model\n\n\n\n\n\n","category":"type"},{"location":"manual/estimators/#2.-Specify-training-parameters","page":"Estimators","title":"2. Specify training parameters","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"TrainingArg","category":"page"},{"location":"manual/estimators/#Microstructure.TrainingArg","page":"Estimators","title":"Microstructure.TrainingArg","text":"TrainingArg(batchsize, lossf, lr, epoch, tv_split, patience )\n\nContain fields related to how network will be trained\n\n\n\n\n\n","category":"type"},{"location":"manual/estimators/#3.-Prepare-network-and-data-for-training","page":"Estimators","title":"3. Prepare network and data for training","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"prepare_training","category":"page"},{"location":"manual/estimators/#Microstructure.prepare_training","page":"Estimators","title":"Microstructure.prepare_training","text":"prepare_trainning(arg::NetworkArg)\n\nreturn mlp,inputs,labels,gt mlp is the network model; inputs and labels are arrays of signals and scaled tissue parameters; gt is a dict containing the ground truth tissue parameters without scaling\n\n\n\n\n\n","category":"function"},{"location":"manual/estimators/#4.-Training-on-generated-training-samples","page":"Estimators","title":"4. Training on generated training samples","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"train_loop!","category":"page"},{"location":"manual/estimators/#Microstructure.train_loop!","page":"Estimators","title":"Microstructure.train_loop!","text":"train_loop!(mlp, trainingarg, inputs, labels)\n\nTrain and update the mlp and return training logs\n\n\n\n\n\n","category":"function"},{"location":"manual/estimators/#5.-test-on-you-data","page":"Estimators","title":"5. test on you data","text":"","category":"section"},{"location":"manual/estimators/","page":"Estimators","title":"Estimators","text":"test","category":"page"},{"location":"manual/estimators/#Microstructure.test","page":"Estimators","title":"Microstructure.test","text":"test(mlp, data, ntest)\n\napply trained mlp to test data ntest times to get mean and std of estimates\n\n\n\n\n\n","category":"function"},{"location":"manual/multithreads/#Multi-threads","page":"Multi threads","title":"Multi threads","text":"","category":"section"},{"location":"manual/multithreads/","page":"Multi threads","title":"Multi threads","text":"Constructing...","category":"page"},{"location":"tutorials/3_data_generation/#How-to-generate-training-datasets","page":"How to generate training datasets","title":"How to generate training datasets","text":"","category":"section"},{"location":"tutorials/3_data_generation/","page":"How to generate training datasets","title":"How to generate training datasets","text":"Constructing...","category":"page"},{"location":"getting_started/#Minimal-steps","page":"Getting started","title":"Minimal steps","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Here includes the minimal steps for you to get started with your MRI dataset. Visit tutorial and manual pages for more feature demonstrations. ","category":"page"},{"location":"getting_started/#Start-julia-in-terminal-with-multi-threads","page":"Getting started","title":"Start julia in terminal with multi-threads","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"~ % julia --threads auto","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"You can also set enviornment variable by adding export JULIA_NUM_THREADS=auto in your bash profile, which will use multi-threads automatically when you start julia.","category":"page"},{"location":"getting_started/#Load-the-package-in-Julia","page":"Getting started","title":"Load the package in Julia","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"In you julia script or REPL:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"julia> using Microstructure","category":"page"},{"location":"getting_started/#Read-dMRI-data-and-perform-spherical-mean","page":"Getting started","title":"Read dMRI data and perform spherical mean","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Provide full path to dMRI images and names of acquisition files with following extensions:     dwiname.bvals, dwiname.bvecs, dwiname.techo, dwiname.tdelta, dwiname.tsmalldel      provide all or a subset of the files depending on the data and model you use. ","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"julia> (dMRIdata, protocol) = spherical_mean(infile_image, normalize=true, save=true, dwiname.bvals, dwiname.bvecs, dwiname.techo, dwiname.tdelta, dwiname.tsmalldel)","category":"page"},{"location":"getting_started/#Specify-the-model-we-want-to-use","page":"Getting started","title":"Specify the model we want to use","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Take MTE-SANDI with a Gaussian noise model for example:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"julia> tissue_model = MTE_SANDI()\njulia> noise_model = Noise_model()","category":"page"},{"location":"getting_started/#Estimation","page":"Getting started","title":"Estimation","text":"","category":"section"},{"location":"manual/compartments/#Tissue-Compartments","page":"Tissue Compartments","title":"Tissue Compartments","text":"","category":"section"},{"location":"manual/compartments/#Abstract-compartment-type","page":"Tissue Compartments","title":"Abstract compartment type","text":"","category":"section"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"Compartment","category":"page"},{"location":"manual/compartments/#Microstructure.Compartment","page":"Tissue Compartments","title":"Microstructure.Compartment","text":"Tissue compartments belong to Compartment type. A compartment contains relevant tissue parameters that affect MRI signals\n\n\n\n\n\n","category":"type"},{"location":"manual/compartments/#axonal-and-dendritic-compartments","page":"Tissue Compartments","title":"axonal and dendritic compartments","text":"","category":"section"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"Cylinder","category":"page"},{"location":"manual/compartments/#Microstructure.Cylinder","page":"Tissue Compartments","title":"Microstructure.Cylinder","text":"Cylinder(da,dpara,d0,t2)\n\nCylinder model using Van Gelderen, P. parameters: cylinder diameter 'da'             parallel diffusivity 'dpara'             intrinsic diffusivity 'd0'             T2 relaxation time 't2'\n\n\n\n\n\n","category":"type"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"Stick","category":"page"},{"location":"manual/compartments/#Microstructure.Stick","page":"Tissue Compartments","title":"Microstructure.Stick","text":"Stick(dpara,t2)\n\nstick model with zero perpendicular diffusivity parameters:  parallel diffusivity 'dpara'             T2 relaxation time 't2'\n\n\n\n\n\n","category":"type"},{"location":"manual/compartments/#extra-cellular-compartment","page":"Tissue Compartments","title":"extra-cellular compartment","text":"","category":"section"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"Zeppelin","category":"page"},{"location":"manual/compartments/#Microstructure.Zeppelin","page":"Tissue Compartments","title":"Microstructure.Zeppelin","text":"Zeppelin(dpara,dperp_frac,t2)\n\nzeppelin/tensor model parameters: parallel diffusivity 'dpara'             perpendicular diffusivity represented as a fraction of dpara 'dperp_frac'             T2 relaxation time 't2'\n\n\n\n\n\n","category":"type"},{"location":"manual/compartments/#cell-body-compartment","page":"Tissue Compartments","title":"cell body compartment","text":"","category":"section"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"Sphere","category":"page"},{"location":"manual/compartments/#Microstructure.Sphere","page":"Tissue Compartments","title":"Microstructure.Sphere","text":"Sphere(diff,size,t2)\n\nsphere model (Neuman) parameters: diffusivity within sphere 'diff'             sphere radius 'size'             T2 relaxation time 't2'\n\n\n\n\n\n","category":"type"},{"location":"manual/compartments/#CSF-and-dot-compartment","page":"Tissue Compartments","title":"CSF and dot compartment","text":"","category":"section"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"Iso","category":"page"},{"location":"manual/compartments/#Microstructure.Iso","page":"Tissue Compartments","title":"Microstructure.Iso","text":"Iso(diff,t2)\n\ndot/isotropic tensor parameters: diffusivity 'diff'             T2 relaxation time 't2' This compartment can be used to represent CSF (diff = free water) or dot compartment (diff = 0).  The latter is for immobile typically seen in ex vivo tissue\n\n\n\n\n\n","category":"type"},{"location":"manual/compartments/#Compartment-signals","page":"Tissue Compartments","title":"Compartment signals","text":"","category":"section"},{"location":"manual/compartments/","page":"Tissue Compartments","title":"Tissue Compartments","text":"compartment_signals","category":"page"},{"location":"tutorials/4_noise_propagation/#How-to-evaluate-accuracy-and-precision","page":"How to evaluate accuracy and precision","title":"How to evaluate accuracy and precision","text":"","category":"section"},{"location":"tutorials/4_noise_propagation/","page":"How to evaluate accuracy and precision","title":"How to evaluate accuracy and precision","text":"Constructing...","category":"page"},{"location":"#Microstructure.jl","page":"Home","title":"Microstructure.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Microstructure.jl is a Julia toolbox aiming at fast and probabilistic microstructure imaging. It supports flexible and extendable compartment modelling with diffusion MRI and combined diffusion-relaxometry MRI. ","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install Microstructure.jl, type ] in Julia to enter package mode and add the package","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]\n(@v1.8) pkg> add Microstructrue","category":"page"},{"location":"","page":"Home","title":"Home","text":"or use github link to keep up to date:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(@v1.8) pkg> add https://github.com/Tinggong/Microstructure.jl.git","category":"page"},{"location":"#Feature-Summary","page":"Home","title":"Feature Summary","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"combined diffusion-relaxometry compartment modelling\nflexible in creating models and adjusting assumptions\ngeneric mcmc estimator\nparallel computing \nquality checking  ","category":"page"},{"location":"#Relationship-to-Other-Packages","page":"Home","title":"Relationship to Other Packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out FreeSurfer.jl. Microstructure.jl also uses I/O functions from FreeSurfer.jl for reading and writing mri image files. ","category":"page"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"}]
}