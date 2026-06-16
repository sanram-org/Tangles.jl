# The `TensorNetwork` class

```@setup plot
using Tenet
using EinExprs
using Makie
Makie.inline!(true)
set_theme!(resolution=(800,400))
using GraphMakie
using CairoMakie
CairoMakie.activate!(type = "svg")
using NetworkLayout
```

In `Tenet`, Tensor Networks are represented by the [`TensorNetwork`](@ref) type.
In order to fit all posible use-cases of [`TensorNetwork`](@ref) implements a **hypergraph**[^2] of [`Tensor`](@ref) objects, with support for open-indices and multiple shared indices between two tensors.

[^2]: A hypergraph is the generalization of a graph but where edges are not restricted to connect 2 vertices, but any number of vertices.

For example, this Tensor Network...

```@raw html
<img class="light-only" width="70%" src="/assets/tn-sketch-light.svg" alt="Sketch of a Tensor Network"/>
<img class="dark-only" width="70%" src="/assets/tn-sketch-dark.svg" alt="Sketch of a Tensor Network (dark mode)"/>
```

... can be constructed as follows:

```@repl plot
tn = TensorNetwork([
    NamedTensor(zeros(2,2), (:i, :m)), # A
    NamedTensor(zeros(2,2,2), (:i, :j, :p)), # B
    NamedTensor(zeros(2,2,2), (:n, :j, :k)), # C
    NamedTensor(zeros(2,2,2), (:p, :k, :l)), # D
    NamedTensor(zeros(2,2,2), (:m, :n, :o)), # E
    NamedTensor(zeros(2,2), (:o, :l)), # F
])
```

[`Tensor`](@ref)s can be added or removed after construction using [`push!`](@ref), [`pop!`](@ref), [`delete!`](@ref) and [`append!`](@ref) methods.

```@repl plot
A = only(pop!(tn, [:i, :m]))
tn
push!(tn, A)
```

You can also replace existing tensors and indices with [`replace`](@ref) and [`replace!`](@ref).

```@repl plot
:i ∈ tn
replace!(tn, :i => :my_index)
:i ∈ tn
:my_index ∈ tn
replace!(tn, :my_index => :i) # hide
```

!!! warning
    Note that although it is a bit unusual but completely legal to have more than one tensor with the same indices, there can be problems when deciding which tensor to be replaced.
    Because of that, you **must** pass the exact tensor you want to replace. A copy of it won't be valid.

## The `AbstractTensorNetwork` interface

Subclasses of [`TensorNetwork`](@ref) inherit from the [`AbstractTensorNetwork`](@ref Tenet.AbstractTensorNetwork) abstract type.
Subtypes of it are required to implement a [`TensorNetwork`](@ref) method that returns the composed [`TensorNetwork`](@ref) object.
In exchange, [`AbstracTensorNetwork`](@ref Tenet.AbstractTensorNetwork) automatically implements [`tensors`](@ref) and [`inds`](@ref) methods for any interface-fulfilling subtype.

As the names suggest, [`tensors`](@ref) returns tensors and [`inds`](@ref) returns indices.

```@repl plot
tensors(tn)
inds(tn)
```

What is interesting about them is that they implement a small query system based on keyword dispatching. For example, you can get the tensors that contain or intersect with a subset of indices using the `contains` or `intersects` keyword arguments:

!!! note
    Keyword dispatching doesn't work with multiple unrelated keywords. Checkout [Keyword dispatch](@ref) for more information.

```@repl plot
tensors(tn; contains=[:i,:m]) # A
tensors(tn; intersects=[:i,:m]) # A, B, E
```

Or get the list of open indices (which in this case is none):

```@repl plot
inds(tn; set = :open)
```

The list of available keywords depends on the layer, so don't forget to check the 🧭 API reference!

## Contraction

When contracting two tensors in a Tensor Network, diagrammatically it is equivalent to fusing the two vertices of the involved tensors.

```@raw html
<figure>
<img class="light-only" width="70%" src="/assets/tensor-matmul-light.svg" alt="Matrix Multiplication using Tensor Network notation"/>
<img class="dark-only" width="70%" src="/assets/tensor-matmul-dark.svg" alt="Matrix Multiplication using Tensor Network notation (dark mode)"/>
<figcaption>Matrix Multiplication using Tensor Network notation</figcaption>
</figure>
```

The ultimate goal of Tensor Networks is to compose tensor contractions until you get the final result tensor.
Tensor contraction is associative, so mathematically the order in which you perform the contractions doesn't matter, but the computational cost depends (and a lot) on the order (which is also known as _contractio path_).
Actually, finding the optimal contraction path is a NP-complete problem and general tensor network contraction is #P-complete.

But don't fear! Optimal contraction paths can be found for small tensor networks (i.e. in the order of of up to 40 indices) in a laptop, and several approximate algorithms are known for obtaining quasi-optimal contraction paths.
In Tenet, contraction path optimization is delegated to the [`EinExprs`](https://github.com/bsc-quantic/EinExprs) library.
A `EinExpr` is a lower-level form of a Tensor Network, in which the contents of the arrays have been left out and the contraction path has been laid out as a tree. It is similar to a symbolic expression (i.e. `Expr`) but in which every node represents an Einstein summation expression (aka `einsum`). You can get the `EinExpr` (which again, represents the contraction path) by calling [`einexpr`](@ref).

```@repl plot
path = einexpr(tn; optimizer=Exhaustive())
```

Once a contraction path is found, you can pass it to the [`contract`](@ref) method. Note that if no contraction `path` is provided, then Tenet will choose an optimizer based on the characteristics of the Tensor Network which will be used for finding the contraction path.

```@repl plot
contract(tn; path=path)
contract(tn)
```

If you want to manually perform the contractions, then you can indicate which index to contract by just passing the index. If you call [`contract!`](@ref), the Tensor Network will be modified in-place and if [`contract`](@ref) is called, a mutated copy will be returned.

```@repl plot
contract(tn, :i)
```

## Visualization

`Tenet` provides visualization support with [`GraphMakie`](https://github.com/MakieOrg/GraphMakie.jl). Import a [`Makie`](https://docs.makie.org/) backend and call [`GraphMakie.graphplot`](@ref) on a [`TensorNetwork`](@ref).

```@example plot
graphplot(tn, layout=Stress(), labels=true)
```
