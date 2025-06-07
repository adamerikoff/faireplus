# faireplus: Comparative Sequence Generation Experiment

faireplus is an experiment designed for **comparing various approaches to sequence generation**. While it initially focuses on French toponyms (place names), its architecture is built to be easily **extensible to other domains and languages**.

---

## Generated Sequences Examples

Below are examples of sequences generated using different configurations and models within this experiment.

### Baseline Generated Sequences (Simple Bigram Probabilities)

These sequences were generated using an initial or default configuration, specifically by **calculating simple bigram probabilities manually**.

```
hievioièran
sa ffoceure-baulle-as
s
drecotesonogarse-rn
gr-mazparçausenese
```

---

### Sequences from Trained Model (Single Layer)

These sequences were generated after training a single-layer model with a Bigram dataset.

```
cuiles-vis-les-dècouresux
mas-les-lalalalay-cay-la-pales-diles-letacoufay
res-bouzis-les-les-les-les-bouyrces-vilates-fcc
bomala-laméppl-lala bsun
balap-les-magnes-les-lay
```

---

### Sequences from Trained Model: Single Embedding Layer, Random Init, Tanh Activation

This section showcases sequences generated from a model trained with specific architectural choices: a single embedding layer, random initialization, and a Tanh activation function.

```
peyrel
montours
champagnère
bas
maint-tcolins-la-primiranche-bylanquie-lès-tourneille
yvelles
esle
chal
med
sallan-de-les
rauent du voie
touly
houties
lynésères
mazet
juen-verneuf
vergent de la boupe
saine-sainvilla
ignac-et-la-villong-sur-sèvres
veur
```

---

### Sequences from Trained Model: 5 Layers, Batch Normalization, ReLU, Kaiming Init

Here are sequences generated from a more complex model configuration, featuring 5 layers, batch normalization, ReLU activation, and Kaiming initialization.

```
départeme
bagne
saint-sur-mer
peyrat-sur-mer
saines
la champas-de-mar
saint-martement de
luy-de-bois-sur-sai
saint-martement
saint-martement des-
```

---

## References

* **Andrej Karpathy's "makemore" series**: This experiment is inspired by and draws concepts from Andrej Karpathy's educational series on building character-level language models.
    * [https://karpathy.ai/](https://karpathy.ai/)