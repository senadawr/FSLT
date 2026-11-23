# Continuous recognition and translation

The available isolated-sign datasets are not sufficient to claim continuous
sentence translation. The continuous subsystem must consume uninterrupted
signing sequences with ordered gloss annotations, optional segment timestamps,
signer/session metadata, and Filipino/English translations. A BiGRU + CTC
model and a separate gloss-to-text model can be added behind the current API.

Until those datasets are trained and evaluated, the product must show the
experimental notice and must not concatenate isolated labels or fabricate
translated sentences.
