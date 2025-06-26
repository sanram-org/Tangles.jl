# Tensor Network

| Required method   | Description |
| :---------------- | :---------- |
| `all_tensors(tn)` |             |
| `Base.copy(tn)`   |             |

## Optional methods

| Method                             | When should this method be defined | Default defintion | Brief description |
| :--------------------------------- | :--------------------------------- | :---------------- | :---------------- |
| `all_tensors_iter(tn)`             |                                    |                   |                   |
| `all_inds(tn)`                     |                                    |                   |                   |
| `hastensor(tn, tensor)`            |                                    |                   |                   |
| `hasind(tn, ind)`                  |                                    |                   |                   |
| `ntensors(tn, tensor)`             |                                    |                   |                   |
| `tensors_with_inds(tn, inds)`      |                                    |                   |                   |
| `tensors_contain_inds(tn, inds)`   |                                    |                   |                   |
| `tensors_intersect_inds(tn, inds)` |                                    |                   |                   |
| `size_inds`                        |                                    |                   |                   |
| `size_ind`                         |                                    |                   |                   |
| `tensor_at`                        |                                    |                   |                   |
| `ind_at`                           |                                    |                   |                   |

## Mutating methods

| Method                          | Description |
| :------------------------------ | :---------- |
| `addtensor!(tn, tensor)`        |             |
| `rmtensor!(tn, tensor)`         |             |
| `replate_tensor!(tn, old, new)` |             |
| `replace_ind!(tn, old, new)`    |             |
