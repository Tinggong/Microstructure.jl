using Documenter, Microstructure

push!(LOAD_PATH, "../src/")

mathengine = MathJax3(
    Dict(
        :loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
        :tex => Dict(
            "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
            "packages" => ["base", "ams", "autoload", "mathtools", "require"],
        ),
    ),
)

makedocs(;
    sitename="Microstructure.jl",
    authors="Ting Gong",
    modules=[Microstructure],
    clean=true,
    doctest=false,
    linkcheck=true,
    warnonly=[:docs_block, :missing_docs, :cross_references, :linkcheck],
    format = Documenter.HTML(;
        mathengine=mathengine,
        sidebar_sitename = false,
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Manual" => Any[
            "manual/dMRI.md",
            "manual/compartments.md",
            "manual/models.md",
            "manual/estimators.md",
            "manual/multithreads.md",
        ],
        "Demos in preprint" => Any[
            "guide/0_intro.md",
            "guide/1_sensitivity_range.md",
            "guide/2_two_stage_MCMC.md",   
            "guide/3_fitting_eval.md", 
            "guide/4_smt.md",   
            "guide/5_sandi.md",                  
        ],        
        "Tutorials" => Any[
            "tutorials/1_build_models.md",
            "tutorials/2_quality_of_fit.md",
            "tutorials/3_data_generation.md",
            "tutorials/4_noise_propagation.md",
            "tutorials/5_model_selection.md",
        ],
        "guide.md",
    ],
)

deploydocs(; repo="github.com/Tinggong/Microstructure.jl.git", push_preview=true)
