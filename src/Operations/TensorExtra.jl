using QuantumTags

# TODO to be moved to a MuscleQuantumTagsExt pkg extension? `plugs` belongs to Tangles as of right now
plugs(tensor::Tensor) = filter!(isplug, map(x -> x.tag, inds(tensor)))

# TODO extend `ind_at` for `Tensor`?
