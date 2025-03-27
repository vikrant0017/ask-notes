# A Recipe for Training Neural Networks

## About

**Author:** Andrej Karpathy  
**Date:** April 25, 2019  

Some few weeks ago I posted a tweet on “the most common neural net mistakes,” listing a few common gotchas related to training neural nets. The tweet got quite a bit more engagement than I anticipated (including a webinar :)). Clearly, a lot of people have personally encountered the large gap between “here is how a convolutional layer works” and “our convnet achieves state-of-the-art results.”

So I thought it could be fun to brush off my dusty blog to expand my tweet into the long form that this topic deserves. Instead of listing common errors, I wanted to dig deeper into how one can avoid making these errors altogether (or fix them very fast). The trick to doing so is to follow a certain process, which as far as I can tell is not very often documented.

Let's start with two important observations that motivate it.

---

## 1) Neural Net Training is a Leaky Abstraction

It is allegedly easy to get started with training neural nets. Numerous libraries and frameworks take pride in displaying 30-line miracle snippets that solve your data problems, giving the (false) impression that this stuff is plug-and-play.

For example:

```python
>>> your_data = # plug your awesome dataset here
>>> model = SuperCrossValidator(SuperDuper.fit, your_data, ResNet50, SGDOptimizer)
# conquer world here
```

These libraries activate the part of our brain familiar with standard software—where clean APIs and abstractions are attainable. Consider the `requests` library:

```python
>>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
>>> r.status_code
200
```

That's cool! The complexity of HTTP requests is hidden behind a few lines of code. But neural nets are not like that. They are not "off-the-shelf" technology once you deviate from standard tasks. Just because you can formulate your problem as reinforcement learning doesn’t mean you should. If you insist on using the technology without understanding how it works, you are likely to fail.

---

## 2) Neural Net Training Fails Silently

When you break or misconfigure regular code, you often get an exception. However, neural nets often fail **silently**. Everything could be correct syntactically, but the network isn’t arranged properly, making it really hard to tell. The “possible error surface” is large, logical (as opposed to syntactic), and very tricky to test.

Examples:
- You forgot to flip your labels when flipping an image during augmentation.
- Your autoregressive model accidentally takes the thing it’s trying to predict as input due to an off-by-one bug.
- You initialized weights from a pretrained checkpoint but didn’t use the original mean.
- You misconfigured regularization strengths, learning rates, decay rates, model size, etc.

Your model will **still train** but might just work slightly worse.

---

## The Recipe

### 1. Become One with the Data

Before touching any neural net code, **inspect your data**:
- Spend hours scanning thousands of examples.
- Look for duplicates, corruptions, biases, and imbalances.
- Understand variations in the data and what features matter.

Run simple scripts to:
- Search/filter/sort by different attributes.
- Visualize distributions and outliers.

Outliers **almost always** reveal data quality issues.

### 2. Set Up the Training/Evaluation Skeleton & Get Dumb Baselines

Start with a simple model (e.g., a linear classifier or tiny ConvNet) to set up the pipeline. **Do not jump straight into complex architectures.**

#### Tips & Tricks:
- **Fix random seed.** Guarantee reproducibility.
- **Disable unnecessary features.** No data augmentation at this stage.
- **Ensure correct loss at initialization.**
- **Use human interpretable metrics.** Track accuracy, F1-score, etc.
- **Overfit one batch.** Your network should memorize a tiny batch.
- **Visualize inputs before they enter the network.**
- **Verify prediction dynamics.** Track how outputs change over training.
- **Use backpropagation to chart dependencies.** Ensure no unintended information leakage.

### 3. Overfit

Once the pipeline works, try to **overfit a model**:
1. Choose an architecture that can overfit.
2. Verify that you can drive training loss to near zero.

#### Tips & Tricks:
- **Copy-paste architectures from relevant papers.** Don’t be a hero.
- **Use Adam (lr=3e-4) at the start.** It’s more forgiving.
- **Introduce complexity gradually.**
- **Disable learning rate decay at first.** Tune this later.

### 4. Regularize

If the model overfits, **regularize it to improve validation accuracy**.

#### Tips & Tricks:
- **Get more data.** The best way to improve generalization.
- **Use aggressive data augmentation.**
- **Consider synthetic data.** GANs, domain randomization, etc.
- **Use pretrained models.** Helps in almost all cases.
- **Reduce input dimensionality.** Remove spurious features.
- **Reduce model size.** Prevent overfitting with a smaller model.
- **Use dropout and weight decay carefully.**
- **Use early stopping.** Stop training when validation loss stops improving.

### 5. Tune

Once your model generalizes well, fine-tune it.

#### Tips & Tricks:
- **Use random search over grid search.** Hyperparameter sensitivity varies.
- **Consider Bayesian optimization tools.** But manual tuning is often best.

### 6. Squeeze Out the Juice

To get the last bits of performance:
- **Use ensembles.** Gain ~2% accuracy in most cases.
- **Let it train longer.** Some models take longer to converge.

---

## Conclusion

Once you reach this point, you’ll have all the ingredients for success:
- **Deep understanding of your dataset**
- **Trustworthy training/evaluation pipeline**
- **Iterative improvements with justified complexity**

Now you’re ready to **read more papers, try more experiments, and push towards state-of-the-art results**. Good luck!
