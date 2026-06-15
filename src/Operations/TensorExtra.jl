plugs(tensor::Tensor) = filter!(isplug, map(x -> x.tag, inds(tensor)))

# TODO extend `ind_at` for `Tensor`?
