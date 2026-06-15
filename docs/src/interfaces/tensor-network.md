# Tensor Network interface

| Required method   | Description |
| :---------------- | :---------- |
| `Base.copy(tn)`   |             |
| `all_tensors(tn)` |             |
| `all_inds(tn)`    |             |

## Optional methods

| Method                            | Brief description |
| :-------------------------------- | :---------------- |
| `all_tensors_iter(tn)`            |                   |
| `all_inds_iter(tn)`               |                   |
| `hastensor(tn, tensor)`           |                   |
| `hasind(tn, ind)`                 |                   |
| `ntensors_all(tn)`                |                   |
| `tensors_set_equal(tn, inds)`     |                   |
| `tensors_set_contain(tn, inds)`   |                   |
| `tensors_set_intersect(tn, inds)` |                   |
| `size_inds`                       |                   |
| `size_ind`                        |                   |

## Mutating methods

| Method                          | Description |
| :------------------------------ | :---------- |
| `addtensor!(tn, tensor)`        |             |
| `rmtensor!(tn, tensor)`         |             |
| `replate_tensor!(tn, old, new)` |             |
| `replace_ind!(tn, old, new)`    |             |
| `slice!(tn, ind, i)`            |             |
| `fuse!(tn, inds)`               |             |

## Dispatching methods

These methods are "syntactic sugar" for end users.

!!! warning

    You SHOULD NOT specialize these methods, as they do not delegate their execution.

| Method                     | Calling method                   |
| :------------------------- | :------------------------------- |
| `tensors(tn)`              | `all_tensors(tn)`                |
| `tensors(tn; contain=i)`   | `tensors_set_contain(tn, i)`     |
| `tensors(tn; intersect=i)` | `tensors_set_intersect(tn, i)`   |
| `tensors(tn; equal=i)`     | `tensors_set_equal(tn, i)`       |
| `tensor(tn; kwargs...)`    | `only(tensors(tn; kwargs...))`   |
| `tensor(tn; at=tag)`       | `tensor_at(tn, tag)`             |
| `ntensors(tn; kwargs...)`  | `length(tensors(tn; kwargs...))` |
| `ntensors(tn)`             | `ntensors_all(tn)`               |
| `inds(tn; set=:all)`       | `all_inds(tn)`                   |
| `inds(tn; set=:open)`      | `inds_set_open(tn)`              |
| `inds(tn; set=:inner)`     | `inds_set_inner(tn)`             |
| `inds(tn; set=:hyper)`     | `inds_set_hyper(tn)`             |
| `ind(tn; kwargs...)`       | `only(inds(tn; kwargs...))`      |
| `ind(tn; at=tag)`          | `ind_at(tn, tag)`                |
| `ninds(tn; kwargs...)`     | `length(inds(tn; kwargs...))`    |
| `ninds(tn)`                | `ninds_all(tn)`                  |

## Extra methods

| Method                      | Brief description |
| :-------------------------- | :---------------- |
| `arrays(tn; kwargs...)`     |                   |
| `contract(tn; kwargs...)`   |                   |
| `resetinds!(tn; kwargs...)` |                   |
