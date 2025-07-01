module TanglesGraphRecipesExt

using Tangles
using Plots
using Graphs
using GraphRecipes
import GraphRecipes: graphplot

function tinycircle(x, y, nodeheight, nodewidth)
    npoints = 10   # was: 100, number of points for smoothness
    θ = range(0, 2π; length=npoints)
    r_x = nodewidth / 20
    r_y = nodeheight / 20
    [(x + r_x * cos(t), y + r_y * sin(t)) for t in θ]
end

function GraphRecipes.graphplot(
    tn::Tangles.AbstractTensorNetwork;
    node_labels=false,
    inner_edge_labels=false,
    open_edge_labels=false,
    curves=false,
    nodeshape=:circle,
    nodesize=0.2,
    kwargs...,
)
    if !isempty(inds(tn; set=:hyper))
        throw(ArgumentError("hyper indices not supported for visualization yet"))
    end

    tensormap = IdDict(tensor => i for (i, tensor) in enumerate(tensors(tn)))

    g = Graphs.SimpleGraph(ntensors(tn))

    #elabels = Array{String}(undef, ntensors(tn),ntensors(tn))
    num_nodes = ntensors(tn)
    num_labels = num_nodes + length(inds(tn; set=:open))
    elabels = fill("", num_labels, num_labels)

    # Add edges between contracted tensors 
    for ii in inds(tn; set=:inner)
        edge_tensors = tensors(tn; intersect=ii)

        @assert length(edge_tensors) == 2
        a, b = edge_tensors

        add_edge!(g, tensormap[a], tensormap[b])
        if inner_edge_labels
            # symmetrize by hand 
            elabels[tensormap[a], tensormap[b]] = string(ii.tag)
            elabels[tensormap[b], tensormap[a]] = string(ii.tag)
        end
    end

    # Add ghost nodes for open indices at the end 
    ghostnodes = Int[]
    for ii in inds(tn; set=:open)
        add_vertex!(g)
        ghost_node = nv(g)
        push!(ghostnodes, ghost_node)
        for _tensor in tensors(tn; intersect=ii)
            add_edge!(g, ghost_node, tensormap[_tensor])
            if open_edge_labels
                # symmetrize by hand 
                elabels[ghost_node, tensormap[_tensor]] = string(ii.tag)
                elabels[tensormap[_tensor], ghost_node] = string(ii.tag)
            end
        end
    end

    nlabels = []
    # Node labels
    if node_labels
        nlabels = [string(i) for i in 1:nv(g)]
    end

    # Node colors: ghost nodes in black, others in a color
    ncolors = [i in ghostnodes ? "black" : "orange" for i in 1:nv(g)]

    # Node sizes: ghost nodes small
    #nsizes = [i in ghostnodes ? 0.001 : 1 for i in 1:nv(g)]

    nshapes = [i in ghostnodes ? tinycircle : :circle for i in 1:nv(g)]
    # @info g
    # @info ne(g)
    # @info nv(g)
    # @info elabels
    # @info adjacency_matrix(g)

    plt = graphplot(
        g;
        nodesize,
        curves,
        nodeshape=nshapes,
        names=nlabels,
        markercolor=ncolors,
        #node_weights=nsizes,
        #markersize=nsizes,
        edgelabel=elabels,
        kwargs...,
    )

    return plt
end

end
