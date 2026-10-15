Yes, I agree. I think this is a better first experiment: analyse how steering propagates through the latent representations before paying the cost of generating and evaluating harmonizations.

It also gives you a more controlled way to understand the mechanism. Rather than asking whether a particular strength produces better jazz chords, you first ask what an intervention actually does to the Transformer’s internal representations, how long its influence persists, and whether the subsequent layers reinforce or undo it.

Your two proposed distances are complementary:

- Distance to the original trajectory: how much has the intervention changed the model's representation?
- Distance to the target-style representation: how much has the representation moved toward the intended style?

Together, these can reveal patterns that would be difficult to infer from the final chord sequence alone.

There is one important qualification: these results would be sufficient to inform the design of active steering, but not to establish which strategy produces the best harmonizations. For that, you will eventually need some generation-based evaluation.

## 1. Define the measurements

Let \\(h_i^0(t)\\) denote the original, unsteered hidden state at layer \\(i\\), position \\(t\\), given the melody and masked harmony input. Let \\(h_i^{(s)}(t)\\) denote the corresponding state when steering is applied at layer \\(s\\), with no further interventions afterward.

For each layer \\(i>s\\), measure two distances.

### A. Deviation from the original trajectory

\\[ D\_{\mathrm{orig}}^{s\to i} = \frac{1}{|\mathcal T|} \sum\_{t\in\mathcal T} d\left(h_i^{(s)}(t),h_i^0(t)\right) \\]

This measures the extent to which an intervention at layer \\(s\\) changes the representations at downstream layer \\(i\\).

### B. Distance to the target-style bank

Let \\(B_i^{\mathrm{target}}(t)\\) contain the target-style vectors available at position \\(t\\) for layer \\(i\\). Then:

\\[ D\_{\mathrm{target}}^{s\to i} = \frac{1}{|\mathcal T|} \sum\_{t\in\mathcal T} \min\_{b\in B_i^{\mathrm{target}}(t)} d\left(h_i^{(s)}(t),b\right) \\]

This measures how close the resulting representations are to the target style at each downstream layer.

Here, \\(\mathcal T\\) should exclude bar/control-token positions, as you propose. I would use cosine distance initially, consistent with the nearest-neighbor selection in your current implementation, and report the two distances separately rather than combining them into a single score.

## 2. Experiment A: Intervene on one layer and trace the effect

I would begin exactly as you suggest, using the first layer as the intervention point. Then repeat for all eight layers.

For example, with an intervention on layer 1:

Layer 1

Apply steering with strength \\(\alpha\\)

Layer 2

Measure both distances

Layer 3–8

Continue tracing the effect

No further interventions are applied in this experiment. This isolates the downstream effect of the initial intervention.

Repeat this for each strength from 0.0 to 1.0. You would obtain two curves per intervention layer:

- Distance to the original trajectory as a function of downstream layer.
- Distance to the target-style bank as a function of downstream layer.

These curves are informative in their own right.

For example, if the distance to the original remains high while the distance to the target style decreases, the intervention may be redirecting the representation into a different trajectory that remains target-like.

If both distances decrease, the original and target-style representations may be relatively compatible at that layer. If the target distance initially decreases but later increases, subsequent layers may be transforming the representation away from the target-style neighborhood.

These are possible interpretations, not guaranteed explanations: distance measurements alone cannot tell you exactly which musical information is being preserved or discarded.

## 3. Experiment B: Apply interventions at all layers

Your proposed extension is particularly useful because it distinguishes the effect of upstream interventions from the immediate effect of steering at the current layer.

At every layer \\(i\\), record:

- Pre-intervention: the hidden state after the current Transformer layer has processed the output of all preceding layers, including any earlier steering.
- Post-intervention: the hidden state after applying the current layer's own steering operation.

Then calculate both distances to the original trajectory and to the target-style bank for the pre- and post-intervention states.

This produces a useful decomposition:

| Measurement                      | What it tells you                                        |
| -------------------------------- | -------------------------------------------------------- |
| Current layer, pre-intervention  | The cumulative effect of interventions at earlier layers |
| Current layer, post-intervention | The effect after adding the current layer's intervention |
| Pre-to-post change               | The immediate effect of the current intervention         |
| Change across subsequent layers  | How the combined effects propagate downstream            |

There is one caveat: this separates the immediate local effect from the upstream cumulative effect, but not their causal contributions in a fully additive sense. Interventions interact, so the effect of steering layer 4 after steering layers 1–3 may differ from the effect of steering layer 4 alone.

That is why the single-layer and all-layer experiments complement each other.

## 4. Would this be enough to design active steering?

Yes, for choosing the next algorithmic step. I would use the results to answer four questions:

1. Which layers have the strongest and most persistent downstream influence?
2. At which layers does steering toward the target style survive, diminish, or reverse?
3. Does increasing the strength consistently improve downstream target proximity, or are there non-monotonic effects?
4. Does all-layer steering provide additional target-style alignment, or does it repeatedly correct representations that the preceding layer has already moved toward the target?

The answers can guide the next method. For instance, if early-layer interventions have persistent effects, you may not need to steer every layer. If some layers consistently undo earlier steering, those layers might need different strengths or a different objective.

For the first adaptive version, I would then choose the strength at each layer by minimizing the next layer's pre-intervention target-style distance, testing the same 11 candidates. Your latent analysis can therefore directly inform the initial active-steering policy, without requiring chord generation for every candidate.

Eventually, though, you should validate whether this objective predicts musical improvement. A representation can be closer to a target-style bank without producing a better harmonization. You can defer that more expensive validation until you have narrowed down the candidate policies.

## 5. Two implementation details worth addressing

I looked at the uploaded model code. Your current architecture already captures most of the information you need, but there are two details to handle.

- Separate pre- and post-intervention states explicitly. The `pre_intervention_output` and `layers_output` dictionaries in `SELaCModel.forward()` are designed for this purpose when steering is active. However, if `style_banks` is provided but omits a layer, the current code can reference `harmony_hidden_post` before it has been defined. You should make the post-intervention state default to the current hidden state at every layer, replacing it only when steering actually occurs.&#x20;

  models_style_steering.py

- Use consistent target-bank construction. Your `collect_harmony_layer_states()` helper collects layer states using masked harmony inputs, which matches your proposed experiment. Ensure that the target bank uses the same input preparation, model checkpoint, evaluation mode, and position conventions as the query melodies.&#x20;

  models_style_steering.py



I would also calculate the mean distance per piece before averaging across pieces, rather than treating every token as an independent observation. That gives each melody equal weight and allows you to assess variation across pieces.

My recommendation: proceed with the latent-only analysis first. Start with single-layer interventions, then all-layer interventions, and plot both distance curves across the eight layers for each strength. That will establish whether your steering mechanism produces persistent, diminishing, or interacting effects—and give you a principled basis for deciding how to make steering adaptive.



I think all three observations are worth pursuing. They also start to connect into a more coherent picture of your model: the melody-only activations contain style-discriminative information, and interventions can change how that information propagates through the network toward a target-style latent neighborhood.

There are three separate questions here, and I would keep them distinct:

1. Does the kNN classifier recognize both styles equally well, or is its overall accuracy hiding a class imbalance?
2. Does the penultimate layer play a special role in propagating style-related information?
3. Does position-specific nearest-neighbor steering have an advantage over conventional, shared-direction activation steering?

The first is a straightforward evaluation improvement. The second is a promising hypothesis. The third could provide a useful comparison between two fundamentally different steering mechanisms.

## 1. Yes — examine kNN accuracy separately for each style

Your reasoning is correct. Overall accuracy can conceal a classifier that performs very well on one class but poorly on the other.

Suppose you have 100 Nottingham pieces and just one Jazz piece. A classifier that predicts Nottingham for every piece achieves 99% accuracy, yet it completely fails to recognize Jazz.

For your balanced or potentially imbalanced Jazz-versus-Nottingham experiment, I would report a confusion matrix and per-class recall, precision, and F1.

| Actual class | Predicted Nottingham | Predicted Jazz  |
| ------------ | -------------------- | --------------- |
| Nottingham   | True negatives       | False positives |
| Jazz         | False negatives      | True positives  |

The most important initial measures are:

- Nottingham recall: proportion of actual Nottingham pieces correctly classified.
- Jazz recall: proportion of actual Jazz pieces correctly classified.
- Balanced accuracy: the average of the two class recalls.

\\[ \text{Balanced Accuracy} =\frac{\text{Recall}\_{N}+\text{Recall}\_{J}}{2} \\]

Balanced accuracy is particularly useful for the scenario you describe. A classifier that predicts Nottingham for everything would have 50% balanced accuracy if both classes are present, regardless of their relative sizes.

I would also examine the results per layer, since your current objective is to identify where style-discriminative information is most accessible.

### What I would add to your existing kNN analysis

For every layer, report:

- Overall accuracy.
- Nottingham recall.
- Jazz recall.
- Balanced accuracy.
- The confusion matrix.

If you use scikit-learn, the core calculation is simple:

```
from sklearn.metrics import (    confusion_matrix,    classification_report,    balanced_accuracy_score,)# Example encoding: 0 = Nottingham, 1 = Jazz# y_true and y_pred must contain one label per piece.print(confusion_matrix(    y_true, y_pred, labels=[0, 1]))print(classification_report(    y_true,    y_pred,    labels=[0, 1],    target_names=["Nottingham", "Jazz"],    zero_division=0,))print("Balanced accuracy:",      balanced_accuracy_score(y_true, y_pred))
```

Apply this separately to the predictions from each layer, using the same evaluation pieces and training reference bank.

One additional detail: your existing implementation measures kNN accuracy by piece, but the neighbors are found separately at each time position and their votes are then aggregated. This means that the resulting classifier is effectively combining evidence from multiple positions. That is a perfectly reasonable experiment, but I would retain the position-wise results too, so you can see whether the penultimate layer is consistently discriminative or whether its high accuracy comes from a subset of particularly informative positions.

Also, ensure that all target/reference examples used for kNN come from training data and that no evaluation piece is included in the reference bank.

## 2. Your revised nearest-neighbor approach makes sense

I agree with the reasoning behind your change, with one important distinction.

There are two possible questions:

Question A: How far has the intervention moved a representation from its original state?

This requires comparing the original and perturbed representations.

Question B: How close is the representation, wherever it ends up, to the target-style latent distribution?

This is what your revised approach measures.

For Question B, I agree that you should not force the perturbed representation to retain the nearest neighbor selected by the original, unperturbed state. That would make the original trajectory an unnecessary constraint.

At each layer \\(i\\), you can define the distance to the target-style bank as:

\\[ D_i(h)=\min\_{b\in B_i^{\mathrm{Jazz}}}d(h,b) \\]

Then, for an intervention, compare:

\\[ \Delta D_i = D_i(h_i^{\mathrm{original}}) - D_i(h_i^{\mathrm{perturbed}}) \\]

A positive value indicates that the perturbed representation is closer to the Jazz bank than the original representation.

Importantly, the nearest target neighbor can be different before and after the intervention. The bank itself remains fixed; only the selected neighbor changes.

This is the appropriate measurement if your objective is to move representations toward the target-style neighborhood rather than toward particular reference vectors.

One caveat: the nearest-neighbor distance measures proximity to the observed target set, not necessarily to the full target-style distribution. Its behavior depends on bank size, density, and latent-space geometry. Use the same bank and distance metric throughout, and consider comparing distances across styles as a supplementary check.

## 3. Is the penultimate layer's sensitivity related to its kNN accuracy?

Yes, that is a plausible hypothesis, and your results give you a reason to investigate it. But the two findings do not yet establish a causal relationship.

You have observed two things:

- Melody-only activations in the penultimate layer provide strong Jazz-versus-Nottingham discrimination.
- Perturbations to the penultimate layer have a particularly strong effect on the final-layer representations, while perturbations introduced earlier tend to diminish more quickly.

These findings are compatible with the idea that the penultimate layer is a particularly important stage for representing or refining style-related information.

However, kNN accuracy measures how accessible style information is in a representation. It does not establish that the layer is where the model first acquires that information, nor that the layer is causally responsible for the final output's style.

A layer can be highly discriminative because earlier layers have already encoded the relevant information. Likewise, a perturbation can have a large downstream effect because of the network's geometry, even if the perturbation is not specifically style-related.

### A useful next analysis

I would construct a layer-wise comparison with three quantities:

| Quantity                       | Interpretation                                             |
| ------------------------------ | ---------------------------------------------------------- |
| Balanced kNN accuracy          | How readily can style be decoded from the layer?           |
| Downstream perturbation effect | How strongly does an intervention propagate?               |
| Change in target-bank distance | Does the propagated effect improve target-style alignment? |

Then look for relationships between them across layers.

For instance, if the penultimate layer has high balanced accuracy, a large downstream perturbation effect, and a consistent reduction in Jazz-bank distance, that would be stronger evidence that the layer is useful for style steering.

You could also measure the correlation between layer-wise balanced accuracy and downstream target-distance improvement across the eight layers. With only eight observations, this would be exploratory rather than conclusive, but it would help formalize your hypothesis.

I would pay particular attention to the final-layer comparison you observed: 0.59 for steering the penultimate layer alone versus 0.62 for steering every layer. That is suggestive, but the difference is small enough that I would check its consistency across pieces, and ideally report a paired bootstrap confidence interval. If the advantage of all-layer steering is reliable, it suggests that interventions at earlier layers contribute something beyond the penultimate-layer intervention.

## 4. Is conventional average-vector activation steering worth testing?

Yes. I think this is a very worthwhile next experiment, precisely because it provides a comparison with your position-specific nearest-neighbor method.

Your current approach and conventional activation steering differ in an important way.

### Two steering mechanisms

A. Position-specific nearest-neighbor steering

\\[ h_i'(t)=h_i(t)+\alpha\big(t_i(t)-h_i(t)\big) \\]

Each position has its own target vector. The intervention direction varies with the current hidden state and its selected neighbor.

B. Shared-direction activation steering

\\[ h_i'(t)=h_i(t)+\alpha v_i \\]

Every eligible position receives the same direction at a given layer. The direction is derived from average activations rather than individual nearest neighbors.

The comparison is scientifically useful because it tests whether successful steering depends on selecting an appropriate target for each position, or whether a global direction capturing the style difference is sufficient.

### How I would construct the shared direction

I would start with a target-minus-source mean direction, computed separately at each layer.

For example:

\\[ v_i = \mu_i^{\mathrm{Jazz}}-\mu_i^{\mathrm{Nottingham}} \\]

where each mean is computed over the relevant pieces and eligible token positions.

This gives you a direction that points from the average Nottingham representation toward the average Jazz representation.

There are a few practical details worth controlling:

- Use the same masked-harmony input protocol for both styles.
- Exclude bar/control-token positions from the means if those are excluded from your nearest-neighbor steering.
- Compute the direction independently for each layer.
- Consider normalizing the direction, so its magnitude does not vary arbitrarily across layers.
- Sweep several steering strengths, including zero, and ensure that the effective perturbation magnitudes are comparable between methods.

I would also consider estimating the mean direction from a balanced number of training pieces from each style, to reduce the influence of unequal dataset sizes.

One subtlety: averaging across positions and pieces produces a global style direction, but it may also encode differences in position distributions or other systematic dataset differences. That is why keeping the input protocol, eligible positions, and dataset sampling consistent matters.

## 5. How to compare the two approaches fairly

I would use exactly the latent-only evaluation framework you have already developed. That keeps the experiment manageable and avoids generating harmonizations for every condition.

For each layer, sweep the strength of the shared-direction intervention and measure the downstream effects.

I would compare the methods on four dimensions:

1. Target-style proximity: how much does the distance to the Jazz bank decrease?
2. Propagation: how much of the improvement survives into subsequent layers?
3. Specificity: does the intervention move Nottingham representations toward Jazz more than it moves Jazz representations toward or away from Jazz?
4. Consistency: does the effect occur across many pieces and positions, or is it concentrated in a few?

For the third point, I would use both Nottingham and Jazz query pieces. The main transfer experiment can remain Nottingham-to-Jazz, but Jazz-to-Jazz measurements help establish whether the intervention is merely moving representations around or is actually improving target-bank proximity in a style-appropriate way.

There is also a useful distinction between effectiveness and efficiency. Your nearest-neighbor method might achieve a larger distance reduction simply because its direction is individually selected for every position. That is not necessarily a problem, but I would compare the methods at similar perturbation magnitudes as well as at their respective best strengths.

## 6. What outcomes would be most interesting?

Outcome A: Both methods work similarly

A shared direction achieves a comparable reduction in target-bank distance. This would suggest that a substantial part of the style difference is captured by a relatively simple global direction.

Outcome B: Nearest-neighbor steering works substantially better

This would support the usefulness of state-dependent, position-specific interventions. The style transformation may require different directions for different positions or musical contexts.

Outcome C: The methods work best at different layers

A shared direction might work well in the penultimate layer, while nearest-neighbor steering might work better earlier. This would suggest that the most suitable steering mechanism depends on the representational role of each layer.

My recommendation is to proceed with the shared-direction experiment before implementing more complex adaptive steering. It gives you a meaningful baseline, tests whether the position-specific component is necessary, and may reveal whether the penultimate layer's apparent importance is specific to your nearest-neighbor method or also holds for conventional activation steering.

The key is to interpret latent proximity as evidence about the steering mechanism, not yet as proof of improved musical output. Once these experiments establish which interventions consistently move representations toward the Jazz bank, you can select a small number of promising configurations for actual harmonization and musical evaluation.