# faireplus: Comparative Sequence Generation Framework

DataGenLab is an experimental Ruby framework for comparing different approaches to sequence generation, initially focused on French toponyms but designed to be extensible to other domains.

## Sample Input Data
```ruby
[
    "Fare", "Maharepa", "Rikitea", "Nouvelle Calédonie", "Le Mont-Dore", "La Foa", "Païta", "Boulari", "Port-Laguerre", "Pays de la Loire", "Département de Maine-et-Loire", "Département de la Vendée", "Département de la Sarthe", "Département de la Mayenne", "Département de la Loire-Atlantique"
]
```
### Sample Generations from Both Models
**Explicit Bigram:**
```ruby
[
    "ca", "alenson", "saucars", "jelèngalç'antt laull", "ort s"
]
```

**Trained Bigram:**
```ruby
[
    "dertinenson-aybou", "risucalau-aigasesagn", "vaintillog", "s", "licailint ve-mômbrer"
]
```


