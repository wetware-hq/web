# `vary-cybernetics` — webs and biological neurons

`type: resource.note` · non-normative; `docs/follow-agent-contract.md` governs on any conflict.

This note records why a web is shaped the way it is for a future coupling with living neurons. Everything below is design intent or hypothesis, not evidence.

## Specification

```json { type: notebook.cell }
{
  "id": "docs/vary-cybernetics.md",
  "type": "resource.note",
  "role": "state the cybernetic reading of a web and what would test it",
  "reads": ["docs/follow-agent-contract.md"],
  "writes": [],
  "must": ["label every biological claim as hypothesis"],
  "check": []
}
```

## Graph

- ← [follow-agent-contract.md](follow-agent-contract.md)

## Reading

- **Deep nets.** A web is a fixed-shape tensor (contract §7), so a deep network reads it directly, with no pixel decoder in between.
- **Neurons.** A web is a map from a position in a stimulus to a response at that position, so its form is a receptive-field map. Each first-layer unit is a sinusoid over the image plane with its own orientation, frequency and phase (from `W1` and `b1`). Hypothesis: that basis is a natural code to drive a stimulating electrode array, and to read one back.
- **Feedback.** Encoding is the closed loop: render, compare with the source, adjust the weights, repeat until fidelity stops improving. A neural interface would close the same loop with a recorded response in place of the render.

## Out of scope

- Spiking units, electrode arrays, cultured neurons and any wet-lab protocol.
- Any claim that a web is how a brain stores images.

## Falsification

- A weight-space network trained on webs does no better than the same network trained on raw pixels of the same images.

## Licence

MIT, see `LICENSE`.
