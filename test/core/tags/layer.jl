using Test
using Tangles
using Tangles: site, bond, plug, partition, layers, LayerSite, LayerBond, Layer, InterLayer, isplug

_layer = Layer(:ket)
@test partition(_layer) == _layer

_ilayer = InterLayer(Layer(:ket), Layer(:bra))
@test partition(_ilayer) == _ilayer
@test layers(_ilayer) == (_ilayer.src, _ilayer.dst)

# `LayerSite`
_site = LayerSite(site"1", :ket)
@test site(_site) == site"1"
@test partition(_site) == Layer(:ket)

_site = LayerSite(site"1,2", "bra")
@test site(_site) == site"1,2"
@test partition(_site) == Layer("bra")

@test isequal(LayerSite(site"1", :ket), LayerSite(site"1", :ket))
@test isequal(site(LayerSite(site"1", :ket)), site(LayerSite(site"1", :not_ket)))
@test !isequal(LayerSite(site"1", :ket), LayerSite(site"2", :ket))
@test !isequal(LayerSite(site"1", :ket), LayerSite(site"2", :not_ket))

# `LayerBond`
_bond = LayerBond(bond"1-2", :ket)
@test bond(_bond) == bond"1-2"
@test partition(_bond) == Layer(:ket)

_bond = LayerBond(bond"1-2", "bra")
@test bond(_bond) == bond"1-2"
@test partition(_bond) == Layer("bra")

_bond = LayerBond(bond"1-2", 1)
@test bond(_bond) == bond"1-2"
@test partition(_bond) == Layer(1)

@test isequal(LayerBond(bond"1-2", :ket), LayerBond(bond"1-2", :ket))
@test isequal(LayerBond(bond"1-2", :ket), LayerBond(bond"2-1", :ket))
@test !isequal(LayerBond(bond"1-2", :ket), LayerBond(bond"1-2", :not_ket))

@test isequal(bond(LayerBond(bond"1-2", :ket)), bond(LayerBond(bond"1-2", :ket)))
@test isequal(bond(LayerBond(bond"1-2", :ket)), bond(LayerBond(bond"1-2", :bra)))
@test !isequal(bond(LayerBond(bond"1-2", :ket)), bond(LayerBond(bond"2-3", :ket)))

# `InterLayerBond`
_inter_bond = InterLayerBond(site"1", :ket => :bra)
@test site(_inter_bond) == site"1"
@test sites(_inter_bond) == (LayerSite(site"1", :ket), LayerSite(site"1", :bra))
@test layers(_inter_bond) == (Layer(:ket), Layer(:bra))
@test partition(_inter_bond) == InterLayer(Layer(:ket), Layer(:bra))

## test set-like equivalence for `InterLayer`
@test hash(InterLayer(Layer(:ket), Layer(:bra))) == hash(InterLayer(Layer(:bra), Layer(:ket)))
@test isequal(InterLayer(Layer(:ket), Layer(:bra)), InterLayer(Layer(:bra), Layer(:ket)))

## test equality under `site`
# @test is_site_equal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :ket => :bra))
# @test is_site_equal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :bra => :ket))
# @test is_site_equal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :asdf => :none))

## test set-like equivalence for `InterLayerBond`
@test isequal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :ket => :bra))
@test isequal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :bra => :ket))
@test !isequal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"2", :ket => :bra))
@test !isequal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :ket => :not_bra))
@test !isequal(InterLayerBond(site"1", :ket => :bra), InterLayerBond(site"1", :not_ket => :bra))

@test hash(InterLayerBond(site"1", :ket => :bra)) == hash(InterLayerBond(site"1", :ket => :bra))
@test hash(InterLayerBond(site"1", :ket => :bra)) == hash(InterLayerBond(site"1", :bra => :ket))
@test hash(InterLayerBond(site"1", :ket => :bra)) != hash(InterLayerBond(site"2", :ket => :bra))
@test hash(InterLayerBond(site"1", :ket => :bra)) != hash(InterLayerBond(site"1", :ket => :not_bra))
@test hash(InterLayerBond(site"1", :ket => :bra)) != hash(InterLayerBond(site"1", :not_ket => :bra))

# `LayerPlug`
_plug = LayerPlug(plug"1", :ket)
@test isplug(_plug)
@test plug(_plug) == plug"1"
@test site(_plug) == LayerSite(site"1", :ket)
@test partition(_plug) == Layer(:ket)
@test adjoint(_plug) == LayerPlug(plug"1'", :ket)

_plug = LayerPlug(plug"1'", :ket)
@test isplug(_plug)
@test plug(_plug) == plug"1'"
@test site(_plug) == LayerSite(site"1", :ket)
@test partition(_plug) == Layer(:ket)
@test adjoint(_plug) == LayerPlug(plug"1", :ket)
