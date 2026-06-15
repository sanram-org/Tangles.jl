# Tagged Tensor Network interface

| Required method       | Description |
| :-------------------- | :---------- |
| `all_sites(tn)`       |             |
| `all_links(tn)`       |             |
| `tensor_at(tn, site)` |             |
| `ind_at(tn, link)`    |             |
| `site_at(tn, tensor)` |             |
| `link_at(tn, index)`  |             |

## Optional methods

| Method                     | Brief description |
| :------------------------- | :---------------- |
| `all_sites_iter(tn)`       |                   |
| `all_links_iter(tn)`       |                   |
| `hassite(tn, site)`        |                   |
| `haslink(tn, link)`        |                   |
| `nsites(tn)`               |                   |
| `nlinks(tn)`               |                   |
| `site_incidents(tn, site)` |                   |
| `link_incidents(tn, link)` |                   |
| `neighbor_sites(tn, site)` |                   |
| `neighbor_links(tn, link)` |                   |

These legacy methods are in the process of being reconsidered:

| Method               | Brief description |
| :------------------- | :---------------- |
| `all_bonds(tn)`      |                   |
| `all_plugs(tn)`      |                   |
| `all_bonds_iter(tn)` |                   |
| `all_plugs_iter(tn)` |                   |
| `bond_at(tn, index)` |                   |
| `plug_at(tn, index)` |                   |
| `hasbond(tn, bond)`  |                   |
| `hasplug(tn, plug)`  |                   |
| `nbonds(tn)`         |                   |
| `nplugs(tn)`         |                   |
| `plugs_set_in(tn)`   |                   |
| `plugs_set_out(tn)`  |                   |
| `plugs_set_dual(tn)` |                   |
| `neighbor_bonds(tn)` |                   |

## Mutating methods

## Dispatching methods

These methods are "syntactic sugar" for end users.

!!! warning

    You SHOULD NOT specialize these methods, as they do not delegate their execution.

| Method                | Calling method        |
| :-------------------- | :-------------------- |
| `sites(tn)`           | `all_sites(tn)`       |
| `links(tn)`           | `all_links(tn)`       |
| `site(tn; at=tensor)` | `site_at(tn, tensor)` |
| `link(tn; at=index)`  | `link_at(tn, index)`  |

These legacy methods are to in the process of being reconsidered:

| Method           | Calling method            |
| :--------------- | :------------------------ |
| `bonds(tn)`      | `all_bonds(tn)`           |
| `plugs(tn)`      | `all_plugs(tn)`           |
| `plugs(tn; set)` | `plugs_set(tn, Val(set))` |

## Extra methods

| Method                   | Brief description      |
| :----------------------- | :--------------------- |
| `cart_sites(tn)`         |                        |
| `canonicalize_inds!(tn)` |                        |
| `align!(a, b)`           | (Use `@align! a => b`) |
| `canconnect(a, b)`       |                        |
| `adjoint_plugs!(tn)`     |                        |
