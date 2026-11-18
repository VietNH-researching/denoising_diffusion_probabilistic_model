# Method

## 1. Forward process

An Ornstein–Uhlenbeck SDE noises the data:

$$dX_t = -\tfrac12 X_t\,dt + dW_t, \qquad X_0 \sim p_\text{data}.$$

Its marginal is Gaussian given $X_0$:

$$X_t = \alpha_t X_0 + \beta_t \varepsilon, \qquad \alpha_t = e^{-t/2},\quad
\beta_t = \sqrt{1 - \alpha_t^2},\quad \varepsilon \sim \mathcal N(0, I).$$

For large $t$, $X_t \approx \mathcal N(0, I)$. In code: `ddpm.sde.ou_coefficients`
(β uses `expm1`, accurate for small $t$) and `ou_forward_sample_batch`.

## 2. Denoiser and loss

The network $\varepsilon_\theta(t, x_t)$ predicts the noise and returns the
denoised estimate (`ddpm.net.Denoiser`):

$$\hat x_0 = \frac{x_t - \beta_t\,\varepsilon_\theta(t, x_t)}{\alpha_t}.$$

The loss (`ddpm.loss.weighted_x0_loss`) is

$$\mathcal L = \mathbb E_{t, x_0, \varepsilon}\Big[\big(\tfrac{\alpha_t}{\beta_t}\big)^2
\,\|\hat x_0 - x_0\|^2\Big].$$

Since $\hat x_0 - x_0 = \frac{\beta_t}{\alpha_t}(\varepsilon - \varepsilon_\theta)$, this is
exactly the noise-prediction loss $\mathbb E\|\varepsilon_\theta - \varepsilon\|^2$.
Training times are $t \sim \mathcal U(t_\text{min}, T)$ with $t_\text{min} = 10^{-4}$,
or biased toward small $t$ (`sample_times_exponential`).

## 3. Score

Tweedie's formula gives the score of the noised marginal from $\hat x_0$
(`ddpm.sde.score_from_x0_hat`):

$$\nabla_x \log p_t(x) \approx -\frac{x - \alpha_t \hat x_0}{\beta_t^2}.$$

## 4. Reverse SDE

The time reversal of the forward SDE is

$$dY = \big[\tfrac12 Y + \nabla \log p_t(Y)\big]\,dt + dW,$$

run from $t = T$ to $0$ with $Y_T \sim \mathcal N(0, I)$. `ddpm.sampler.reverse_sde`
integrates it with Euler–Maruyama, stepping from $t$ to $s < t$:

$$y \leftarrow y + \Delta\big(\tfrac12 y + \nabla\log p_t(y)\big) + \sqrt{\Delta}\,\varepsilon,
\qquad \Delta = t - s.$$

The network is evaluated at every grid time except $t = 0$.

## 5. Time grid

`ddpm.times.exponential_linspace(eps, T, n)` gives $n$ times on $[0, T]$, evenly
spaced in $\log(t + \varepsilon)$:

$$t_k = \exp\!\Big(\log\varepsilon + \tfrac{k}{n-1}\log\tfrac{T+\varepsilon}{\varepsilon}\Big) - \varepsilon.$$

Each step is a constant factor larger than the previous one. With
$\varepsilon = 0.1$, $T = 4$, $n = 200$: steps grow from 0.0019 to 0.076, and 37 of
the 199 steps lie in $t \le 0.1$, where the fine structure of the data is
resolved.

## 6. Choosing T

$T$ must be large enough that $X_T \approx \mathcal N(0, I)$, the distribution
sampling starts from. The coordinate with the largest variance $\sigma^2$
decides it: the std of $X_T$ is $\sqrt{e^{-T}\sigma^2 + 1 - e^{-T}}$. A larger
$T$ than needed spends training and sampling steps where the data is already
pure noise, which blurs samples. Rough rule: $e^{-T}\sigma^2 \approx 0.5$, i.e.
$T \approx \ln(2\sigma^2)$; here $\sigma^2 \approx 33$ gives $T \approx 4$. See
[EXPERIMENTS.md](EXPERIMENTS.md).
