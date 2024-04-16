using Documenter, Microstructure

push!(LOAD_PATH,"../src/")

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
                           :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                                        "packages" => [
                                            "base",
                                            "ams",
                                            "autoload",
                                            "mathtools",
                                            "require",
                                        ])))

makedocs(
    sitename="Microstructure.jl",
    authors="Ting Gong",
    modules=[Microstructure],
    clean=true, doctest=false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs, :cross_references, :linkcheck],
    format = Documenter.HTML(mathengine = mathengine,
                             canonical=""),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => Any[
            "tutorials/1_build_models.md",
            "tutorials/2_quality_of_fit.md",
            "tutorials/3_data_generation.md",
            "tutorials/4_noise_propagation.md",
            "tutorials/5_model_selection.md"
        ],
        "Manual" => Any[
            "manual/dMRI.md",
            "manual/compartments.md",
            "manual/models.md",
            "manual/estimators.md",
            "manual/multithreads.md"
        ],
        "guide.md"
    ]
)

deploydocs(
   repo = "github.com/Tinggong/Microstructure.jl.git"
   #push_preview = true
)