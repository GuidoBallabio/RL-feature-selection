## Bound on Action-Value function

We need to find a bound on the Q-function that quantifies the error in approximating it with a subset of the features such that:

$$ Q_{\pi} \approx \hat{Q}_{\pi} $$

for every policy $\pi$, hence looking at the worst case:

$$ \max_{\pi} {\lVert Q_{\pi} - \hat{Q}_{\pi} \rVert} \leq \epsilon $$

where $\epsilon$ is arbitrarily small, as it is assumed that the whole set of features represents exactly the problem with zero error.

The most basic bound would be:

$$ \epsilon = \lvert R - \hat{R} \rvert + \frac{\gamma}{1-\gamma} \lVert P - \hat{P} \rVert_{TV} $$ 

but we would rather use the information theoretic quantities to select the features related to the reward and then the dynamics behind it. Such that:

$$ \epsilon = f(I(Y;X_i | X_{-i})) .$$ 

(actually not clear yet if the features will be considered two at a time:
$$ \forall i \ KL(p(Y|X_{-i},X_i) \parallel p(Y|X_{-i})) $$
or all together recursively:
$$ KL(p(Y_i,Y_{-i}|X_{-i},X_i) \parallel p(Y_{-i}|X_{-i})) $$
)
