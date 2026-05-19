# HARP — Complete Application Walkthrough

This notebook builds the entire HARP (Hybrid Agent Ranking via Personalized PageRank) system from the ground up. Each section explains the *why* before showing the *how*, so by the end you will not only have a working agent router but also understand every line of math powering it.

You can paste each numbered code block into its own Colab cell, or paste the whole thing into one cell — both work. Run the cells top to bottom.

---

## Section 0 — What we are actually building

Imagine you maintain a fleet of twelve AI agents. Some are general assistants, some are specialist coders, one is a web-browsing scaffold, another is a deep-research bot. A user sends you a new task: *"Open this spreadsheet and tell me which region had the highest Q4 revenue."* Which agent should handle it?

You have three pieces of evidence to draw on. First, history: which agents have succeeded on similar tasks before? Second, endorsement: which agents do other agents trust enough to hand work to? Third, semantics: which agents *describe themselves* in a way that matches what the task is asking for?

A naive system uses only one of these signals. A weighted-average system mixes them with hand-tuned coefficients but ignores the graph structure between agents. HARP does something more principled: it treats the agents, skills, and the current task as nodes in a graph, builds three transition matrices (one per signal), combines them convexly so the result remains a valid Markov chain, and runs PageRank-style power iteration with a task-conditional teleport vector. The stationary distribution gives a score per agent, and the highest scorer wins the task.

The math is provably convergent, the implementation runs in seconds on a laptop, and the whole thing is roughly two hundred lines of Python. Let us build it.

---

## Section 1 — Install dependencies

We need four libraries. `sentence-transformers` gives us a fast 384-dimensional embedder for text. `networkx` is used only for the vanilla-PageRank baseline. `scikit-learn` gives us cosine similarity. `matplotlib` and `seaborn` handle the visualizations at the end. The `-q` flag keeps the install output quiet.

```python
# Cell 1: Install dependencies
!pip -q install sentence-transformers networkx scikit-learn seaborn
```

---

## Section 2 — Imports and global configuration

Now we pull in everything we need and seed the random number generators so the results are reproducible. Setting both `numpy` and Python's `random` seeds matters because some sklearn internals use the latter.

```python
# Cell 2: Imports and seeding
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Reproducibility: same numbers every run
np.random.seed(42)
random.seed(42)

# A few global hyperparameters for HARP. We will explain each one as we use it.
ALPHA = 0.85         # Damping factor — same default as Brin & Page (1998)
KAPPA = 10.0         # Temperature for the softmax teleport vector
BETA_P = 0.5         # Weight on the performance transition matrix M^P
BETA_C = 0.3         # Weight on the endorsement transition matrix M^C
BETA_PHI = 0.2       # Weight on the semantic transition matrix M^phi
TOLERANCE = 1e-7     # Convergence threshold for power iteration
MAX_ITER = 200       # Safety cap on power iterations

print("HARP configuration loaded.")
print(f"  alpha={ALPHA}, kappa={KAPPA}, "
      f"beta=({BETA_P}, {BETA_C}, {BETA_PHI})")
```

Notice that the three beta weights sum to one. This is not an accident — it is the constraint that guarantees our combined transition matrix stays column-stochastic (every column sums to one), which in turn is what guarantees the algorithm converges. We will return to this.

---

## Section 3 — Defining the agent universe

In a real deployment you would pull this list from your agent registry. For the tutorial we hand-craft twelve agents that reflect actual public scaffolds and model families seen on the HAL leaderboard for GAIA. Each agent gets a short natural-language description because that is what the semantic similarity signal will operate on.

```python
# Cell 3: The agent universe
AGENT_NAMES = [
    "HAL-Generalist-Sonnet-4.5",
    "HAL-Generalist-GPT-5",
    "HAL-Generalist-Gemini-2.5",
    "HF-OpenDeepResearch-GPT-4o",
    "HF-OpenDeepResearch-Llama3-70B",
    "AutoGen-Coder",
    "MetaGPT-Engineer",
    "CAMEL-Researcher",
    "Voyager-Skill-Agent",
    "GPTSwarm-Swarm",
    "WebArena-Browser",
    "SWE-Agent-Claude",
]

# Each description is what the agent "advertises" itself as.
# The semantic signal will compare these strings to task descriptions.
AGENT_DESC = {
    "HAL-Generalist-Sonnet-4.5":
        "general purpose assistant with web browsing, code execution, and image understanding tools",
    "HAL-Generalist-GPT-5":
        "general purpose assistant with strong reasoning, code, and multi-step planning",
    "HAL-Generalist-Gemini-2.5":
        "general purpose assistant with multimodal vision and long-context reading",
    "HF-OpenDeepResearch-GPT-4o":
        "deep research agent that browses the web and reads PDF documents to answer questions",
    "HF-OpenDeepResearch-Llama3-70B":
        "open source deep research agent for web search and document reading",
    "AutoGen-Coder":
        "python code generation agent with iterative execution and debugging",
    "MetaGPT-Engineer":
        "software engineering multi-agent pipeline for writing and testing code",
    "CAMEL-Researcher":
        "role-playing research assistant for structured analysis and writing",
    "Voyager-Skill-Agent":
        "lifelong skill library agent that composes tools from past experience",
    "GPTSwarm-Swarm":
        "optimizable agent graph that combines several specialists in parallel",
    "WebArena-Browser":
        "specialist web browsing agent that navigates websites and clicks UI elements",
    "SWE-Agent-Claude":
        "software engineering agent for editing code repositories and writing patches",
}

n_agents = len(AGENT_NAMES)
print(f"Loaded {n_agents} agents.")
```

The descriptions are intentionally varied. Some agents brag about being generalists (which will help them rank on broad tasks but hurt them when the task is very specific), and some are tightly specialized (the reverse trade-off). Watching how HARP balances these is half the fun.

---

## Section 4 — Defining the skill vocabulary

Skills are the bridge between agents and tasks. GAIA's annotator metadata uses a free-form `Tools` field; we discretize this into eight skill categories that cover most of what GAIA L1/L2/L3 tasks demand.

```python
# Cell 4: The skill vocabulary
SKILLS = [
    "web_browsing",
    "pdf_reading",
    "python_coding",
    "image_recognition",
    "math_calculation",
    "audio_transcription",
    "spreadsheet_analysis",
    "web_search",
]

SKILL_DESC = {
    "web_browsing":
        "navigating interactive websites, clicking buttons, filling forms, and reading rendered pages",
    "pdf_reading":
        "extracting and understanding text and tables from PDF documents",
    "python_coding":
        "writing, executing, and debugging python programs to solve computational tasks",
    "image_recognition":
        "understanding the content of images, photographs, diagrams, and screenshots",
    "math_calculation":
        "performing arithmetic, algebra, and symbolic mathematics with precision",
    "audio_transcription":
        "transcribing spoken language from audio files into text",
    "spreadsheet_analysis":
        "reading excel and csv files and computing aggregations over rows and columns",
    "web_search":
        "issuing search engine queries and synthesizing information from results",
}

n_skills = len(SKILLS)
print(f"Loaded {n_skills} skills.")
```

You may notice that some skills overlap semantically (web_browsing and web_search are close cousins). That is intentional — it stress-tests whether HARP can still distinguish them, since their performance histories will differ even when their embeddings look similar.

---

## Section 5 — Synthesizing GAIA-style tasks

In a production system you would stream tasks from a queue. For the tutorial we generate thirty synthetic tasks that mimic GAIA's question style and difficulty mix. The official GAIA split is roughly 31% Level 1, 53% Level 2, and 16% Level 3, so we sample levels with those probabilities.

```python
# Cell 5: Synthetic GAIA-style tasks
TASK_TEMPLATES = [
    "Find the population of {place} from the official census PDF and compute its growth rate.",
    "Open this spreadsheet and return the sum of column B for entries in 2024.",
    "Transcribe the attached audio clip and identify the speaker.",
    "Browse the wikipedia page for {place} and extract the founding year.",
    "Run the attached python script and report the final output.",
    "Identify the country shown in this satellite image and look up its capital city.",
    "Search the web for the latest CEO of {place} and find their previous role.",
    "Open the attached PDF and find the date of the third figure caption.",
    "Compute the integral of the expression on page 4 of the document.",
    "Listen to the recording and write a five sentence summary in english.",
]

PLACE_FILLERS = ["Paris", "Brazil", "Tokyo", "Apollo 11", "Marie Curie",
                 "the Eiffel Tower", "OpenAI", "Anthropic", "Mumbai", "Iceland"]

# Sample 30 tasks with GAIA-calibrated level distribution
n_tasks = 30
levels = np.random.choice([1, 2, 3], size=n_tasks, p=[0.31, 0.53, 0.16])

task_rows = []
for i, lvl in enumerate(levels):
    template = np.random.choice(TASK_TEMPLATES)
    place = np.random.choice(PLACE_FILLERS)
    task_rows.append({
        "task_id": f"task_{i:03d}",
        "desc": template.format(place=place) if "{place}" in template else template,
        "level": int(lvl),
    })

TASKS = pd.DataFrame(task_rows)
print(f"Generated {len(TASKS)} tasks.")
print(f"  Level 1 (easy):   {sum(TASKS.level==1)}")
print(f"  Level 2 (medium): {sum(TASKS.level==2)}")
print(f"  Level 3 (hard):   {sum(TASKS.level==3)}")
TASKS.head(5)
```

If you want to use real GAIA later, you can replace the synthesis block with `datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")` once you have access to the gated repository, and pull the `Question` field into the `desc` column. The rest of the pipeline does not care whether the tasks are real or synthetic.

---

## Section 6 — Embedding everything

Sentence embeddings are how we let HARP "read" the descriptions. The `all-MiniLM-L6-v2` model maps any sentence to a 384-dimensional unit vector such that two sentences with similar meaning have a high cosine similarity. We embed all agents, all skills, and all tasks once and reuse the vectors throughout.

```python
# Cell 6: Embed agents, skills, and tasks
print("Loading sentence transformer (first run downloads ~80MB)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# normalize_embeddings=True returns unit vectors so dot product = cosine similarity
agent_emb = embedder.encode(
    [AGENT_DESC[a] for a in AGENT_NAMES],
    normalize_embeddings=True,
)
skill_emb = embedder.encode(
    [SKILL_DESC[s] for s in SKILLS],
    normalize_embeddings=True,
)
task_emb = embedder.encode(
    TASKS["desc"].tolist(),
    normalize_embeddings=True,
)

print(f"Embeddings ready:")
print(f"  agent_emb: {agent_emb.shape}")
print(f"  skill_emb: {skill_emb.shape}")
print(f"  task_emb:  {task_emb.shape}")
```

A useful sanity check at this point is to look at the cosine similarity matrix between agents and skills. You should see, for instance, that "WebArena-Browser" is closest to "web_browsing", and "SWE-Agent-Claude" is closest to "python_coding". If you do not see that, your embedder is broken or your descriptions are too vague.

---

## Section 7 — Simulating an outcome history

This is the most consequential synthetic step. Each agent has a hidden "true skill" vector — its real probability of succeeding at each skill. We draw these from a Beta distribution so they look like realistic empirical success rates rather than uniform noise. Then we simulate a training period: for each training task, four to six agents attempt it, and each succeeds with probability equal to their average skill on the task's required skills.

The point of this construction is to give HARP a *partial* and *noisy* history, which is what real systems have. No agent has tried every task, and even on tasks they have tried, the outcomes are stochastic.

```python
# Cell 7: Simulate the outcome history H
# Each agent has a latent skill profile drawn from Beta(2, 2)
# plus a small Gaussian jitter to break ties.
true_skill = np.clip(
    np.random.beta(2, 2, size=(n_agents, n_skills))
    + 0.15 * np.random.randn(n_agents, n_skills),
    0.02, 0.98,
)

# Each task implicitly requires the 2 skills whose descriptions are
# semantically closest to it. We compute that here.
task_skill_sim = cosine_similarity(task_emb, skill_emb)
task_required_skills = [np.argsort(-row)[:2].tolist() for row in task_skill_sim]

# Split tasks into training (for building the history) and test (for evaluation)
n_train = len(TASKS) // 2
train_idx = list(range(n_train))
test_idx = list(range(n_train, len(TASKS)))

# Simulate which agents attempted which task and whether they succeeded.
# Each task gets 4-6 random attempters.
history_rows = []
for ti in train_idx:
    skills_i = task_required_skills[ti]
    n_attempters = np.random.randint(4, 7)
    attempters = np.random.choice(n_agents, size=n_attempters, replace=False)
    for ai in attempters:
        # Success probability = mean true skill on the required skills
        p_success = float(np.mean(true_skill[ai, skills_i]))
        outcome = int(np.random.rand() < p_success)
        # Attribute the outcome to each required skill
        for si in skills_i:
            history_rows.append({
                "agent": ai,
                "task": ti,
                "skill": si,
                "outcome": outcome,
            })

H = pd.DataFrame(history_rows)
print(f"History H has {len(H)} (agent, task, skill, outcome) tuples.")
print(f"Overall success rate in training: {H.outcome.mean():.2%}")
print(f"Train tasks: {len(train_idx)}, Test tasks: {len(test_idx)}")
```

The output should show an overall success rate somewhere in the 40-60% range. If it is far outside that, your Beta parameters are skewed and the evaluation later will look weird.

---

## Section 8 — Building the performance matrix W^P

Now we convert the raw history into a performance weight per agent-skill pair. The naive choice would be raw success rate, $n^+ / (n^+ + n^-)$, but this has a famous problem: an agent that tried one task and succeeded gets a perfect 1.0, while an agent that tried 100 tasks and succeeded 99 times gets 0.99 — and the system would prefer the lucky one-shot agent. Bayesian smoothing with a Beta(1,1) prior fixes this elegantly.

The smoothed estimate is $w^P_{a,s} = (1 + n^+_{a,s}) / (2 + n^+_{a,s} + n^-_{a,s})$. With zero observations this returns 0.5 — a principled "I don't know" — and as observations accumulate it converges to the empirical rate. This single line is also our cold-start solution for new agents.

```python
# Cell 8: Performance weights with Beta(1,1) smoothing
def beta_smoothed(successes, attempts):
    """
    Beta-smoothed success rate. Returns the posterior mean of theta
    under a Beta(1,1) prior given (successes, attempts).
    With zero attempts this returns 0.5 — the cold-start prior.
    """
    failures = attempts - successes
    return (1 + successes) / (2 + successes + failures)

W_P = np.zeros((n_agents, n_skills))
for ai in range(n_agents):
    for si in range(n_skills):
        # All (agent, skill) outcomes from training history
        subset = H[(H.agent == ai) & (H.skill == si)]
        n_succ = int(subset.outcome.sum())
        n_att = len(subset)
        W_P[ai, si] = beta_smoothed(n_succ, n_att)

print("Performance matrix W^P computed.")
print(f"  shape: {W_P.shape}")
print(f"  min:   {W_P.min():.3f}  (worst agent on hardest skill)")
print(f"  max:   {W_P.max():.3f}  (best agent on easiest skill)")
print(f"  mean:  {W_P.mean():.3f}  (should be near 0.5 for sparse history)")
```

The mean hovering near 0.5 is a sign the prior is doing its job: most cells in the matrix have few observations, so they pull toward the neutral default. The extreme values come from the few agent-skill pairs with dense, consistent history.

---

## Section 9 — Building the endorsement matrix W^C

Endorsements model "agent A trusts agent B enough to hand it work, and that work turned out well." In GAIA traces we do not have direct invocation logs, so we approximate it via co-success: if agents A and B both succeed on the same tasks, that is a soft signal they are useful together. We subtract a small penalty for co-failure to discourage clustering of bad agents.

This is conceptually the same construction EigenTrust (Kamvar et al., 2003) uses for peer-to-peer reputation, just applied to LLM agents instead of file-sharing peers.

```python
# Cell 9: Endorsement weights via co-success
W_C = np.zeros((n_agents, n_agents))

# Reward: pairs of agents who succeed on the same training task
for ti in train_idx:
    succ_agents = H[(H.task == ti) & (H.outcome == 1)].agent.unique()
    for i in succ_agents:
        for j in succ_agents:
            if i != j:
                W_C[i, j] += 1.0

# Penalty: pairs of agents who both fail on the same task
# (clip at zero so we never have negative weights)
for ti in train_idx:
    fail_agents = H[(H.task == ti) & (H.outcome == 0)].agent.unique()
    for i in fail_agents:
        for j in fail_agents:
            if i != j:
                W_C[i, j] = max(W_C[i, j] - 0.3, 0.0)

print("Endorsement matrix W^C computed.")
print(f"  shape: {W_C.shape}")
print(f"  density (non-zero entries): {(W_C > 0).mean():.2%}")
print(f"  most-endorsed agent: {AGENT_NAMES[W_C.sum(axis=0).argmax()]}")
```

The most-endorsed agent printed here is likely one of the generalists, which makes sense — generalists co-appear on the success lists of many tasks because they are tried frequently. In a real deployment with explicit invocation logs, this signal would carry much sharper information.

---

## Section 10 — Column normalization helper

PageRank requires column-stochastic transition matrices: every column sums to one, representing a probability distribution over where the random walker goes next. We will use this normalization repeatedly, so we wrap it in a function with two safety features: it avoids division by zero (with `eps`) and it leaves all-zero columns untouched so we can handle dangling nodes later.

```python
# Cell 10: Column normalization (with safe handling of zero columns)
def col_normalize(W, eps=1e-12):
    """
    Normalize columns of W to sum to 1. Columns whose total is
    essentially zero are returned untouched; the caller is responsible
    for handling those (we apply a uniform-distribution dangling fix later).
    """
    col_sums = W.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums < eps, 1.0, col_sums)
    return W / col_sums

# Quick sanity check
test = col_normalize(np.array([[1.0, 2.0], [3.0, 4.0]]))
assert np.allclose(test.sum(axis=0), 1.0)
print("Column normalization helper ready.")
```

---

## Section 11 — Building the three transition matrices

This is the heart of HARP. We lay out the full state vector $\mathbf{r}$ over $V = \text{agents} \cup \text{skills} \cup \{\text{current task}\}$. The state vector has length $n_{\text{agents}} + n_{\text{skills}} + 1 = 12 + 8 + 1 = 21$. We build three transition matrices of that size, each encoding one signal type.

The performance matrix $\mathbf{M}^P$ has block structure: probability flows back and forth between agents and skills, weighted by $W^P$. The endorsement matrix $\mathbf{M}^C$ has flow only among agents, weighted by $W^C$. The semantic matrix $\mathbf{M}^\phi$ has flow between skills and the current task (and between agents and the current task), weighted by cosine similarity.

The "dangling node fix" handles the edge case where a column has no outgoing edges: we redirect those to a uniform distribution. This is exactly the same trick Brin and Page used in the original PageRank paper to ensure ergodicity.

```python
# Cell 11: Build the three task-conditional transition matrices
def make_transition_matrices(task_idx):
    """
    For a given task, build the three block matrices M^P, M^C, M^phi
    over the state space V = agents ∪ skills ∪ {task}.
    Returns (M_P, M_C, M_phi, A_idx, S_idx, T_idx) where the last three
    are the index ranges for each node group.
    """
    V = n_agents + n_skills + 1
    A_idx = np.arange(0, n_agents)              # rows 0 ... n_agents-1
    S_idx = np.arange(n_agents, n_agents + n_skills)
    T_idx = n_agents + n_skills                 # the single task node

    # ---------- M^P: Agent <-> Skill via performance ----------
    M_P = np.zeros((V, V))
    # Probability of jumping FROM a skill TO an agent: column = skill, row = agent.
    # Higher performance on that skill means higher inflow to that agent.
    P_skill_to_agent = col_normalize(W_P.T)     # shape (n_skills, n_agents)
    P_agent_to_skill = col_normalize(W_P)       # shape (n_agents, n_skills)
    for ai in range(n_agents):
        for si in range(n_skills):
            M_P[A_idx[ai], S_idx[si]] = P_skill_to_agent[si, ai]
            M_P[S_idx[si], A_idx[ai]] = P_agent_to_skill[ai, si]
    # The task node is a sink in M^P (no performance edges touch it)
    M_P[T_idx, T_idx] = 1.0

    # Dangling-node fix: any all-zero column becomes uniform
    zero_cols = M_P.sum(axis=0) < 1e-12
    M_P[:, zero_cols] = 1.0 / V

    # ---------- M^C: Agent <-> Agent via endorsement ----------
    M_C = np.eye(V)                              # default: self-loop everywhere
    C_agent = col_normalize(W_C)
    for i in range(n_agents):
        M_C[A_idx[i], A_idx] = 0.0               # clear the agent row
        M_C[A_idx, A_idx[i]] = C_agent[:, i]     # fill from endorsement
    zero_cols = M_C.sum(axis=0) < 1e-12
    M_C[:, zero_cols] = 1.0 / V

    # ---------- M^phi: semantic edges to/from the task ----------
    M_phi = np.zeros((V, V))
    # Skill-to-task similarity (and back)
    sim_skill_task = (1 + cosine_similarity(
        skill_emb, task_emb[task_idx:task_idx + 1]).ravel()) / 2
    sim_agent_task = (1 + cosine_similarity(
        agent_emb, task_emb[task_idx:task_idx + 1]).ravel()) / 2

    # FROM task -> skills (half the outgoing mass) and -> agents (other half)
    PHI_to_skills = sim_skill_task / max(sim_skill_task.sum(), 1e-12)
    PHI_to_agents = sim_agent_task / max(sim_agent_task.sum(), 1e-12)
    M_phi[S_idx, T_idx] = 0.5 * PHI_to_skills
    M_phi[A_idx, T_idx] = 0.5 * PHI_to_agents

    # FROM skills -> task (all back to task, weighted by similarity)
    M_phi[T_idx, S_idx] = sim_skill_task / (sim_skill_task.sum() + 1e-12)

    zero_cols = M_phi.sum(axis=0) < 1e-12
    M_phi[:, zero_cols] = 1.0 / V

    return M_P, M_C, M_phi, A_idx, S_idx, T_idx

# Quick verification on one task
_M_P, _M_C, _M_phi, *_ = make_transition_matrices(test_idx[0])
assert np.allclose(_M_P.sum(axis=0), 1.0), "M_P columns must sum to 1"
assert np.allclose(_M_C.sum(axis=0), 1.0), "M_C columns must sum to 1"
assert np.allclose(_M_phi.sum(axis=0), 1.0), "M_phi columns must sum to 1"
print("All three transition matrices are column-stochastic. ✓")
```

The assertions at the end are not decorative — they are the mathematical contract that makes HARP converge. If any column failed to sum to one, the random walker would be "leaking" probability mass and the iteration would not have a unique fixed point.

---

## Section 12 — The task-conditional teleport vector

PageRank's teleport vector is what makes the random walker eventually visit every page even if the graph has dead ends. The original PageRank used a uniform teleport. Personalized PageRank uses a node-specific one. Topic-Sensitive PageRank (Haveliwala 2002) used one of sixteen pre-computed topic vectors. HARP generalizes all three: it computes a *continuous* teleport vector from the task's embedding via a softmax with temperature $\kappa$.

A high $\kappa$ makes the teleport sharply concentrated on the most semantically similar agents and skills (good when you trust the embedding). A low $\kappa$ spreads probability more evenly (good when the embedding is unreliable).

```python
# Cell 12: Build the semantic teleport vector p_tau
def teleport_vector(task_idx, kappa=KAPPA):
    """
    Build a probability distribution over V that concentrates mass on
    nodes (agents and skills) whose descriptions are semantically similar
    to the current task. The task node itself gets zero mass.
    """
    V = n_agents + n_skills + 1
    sim_agents = cosine_similarity(agent_emb, task_emb[task_idx:task_idx + 1]).ravel()
    sim_skills = cosine_similarity(skill_emb, task_emb[task_idx:task_idx + 1]).ravel()
    sims = np.concatenate([sim_agents, sim_skills, np.array([0.0])])
    # softmax with temperature
    p = np.exp(kappa * sims)
    p = p / p.sum()
    return p

# Sanity check on the first test task
_p = teleport_vector(test_idx[0])
assert abs(_p.sum() - 1.0) < 1e-9
top_3_agents = np.argsort(-_p[:n_agents])[:3]
print(f"For task: \"{TASKS.iloc[test_idx[0]]['desc'][:70]}...\"")
print("Top 3 agents by semantic teleport mass:")
for rank, ai in enumerate(top_3_agents, 1):
    print(f"  #{rank}: {AGENT_NAMES[ai]:35s}  mass={_p[ai]:.4f}")
```

These top-3 by semantic mass alone are essentially what a "Sim-only" baseline would return. HARP will use this as its *teleport* and then let the random walk shift mass around based on performance and endorsements. Often it agrees with semantics; sometimes it disagrees, and those cases are where HARP earns its keep.

---

## Section 13 — The HARP power iteration

Now we tie everything together. The HARP update rule is the single line $\mathbf{r}_{t+1} = \alpha \mathbf{M}_\tau \mathbf{r}_t + (1 - \alpha) \mathbf{p}_\tau$, where $\mathbf{M}_\tau = \beta_P \mathbf{M}^P + \beta_C \mathbf{M}^C + \beta_\phi \mathbf{M}^\phi$. We initialize $\mathbf{r}_0 = \mathbf{p}_\tau$ (a sensible warm start) and iterate until consecutive states differ by less than the tolerance in $\ell_1$ norm.

The mathematical guarantee, proved in the writeup: this is a contraction with rate $\alpha$, so $k$ iterations give error at most $\alpha^k$. With $\alpha = 0.85$ and tolerance $10^{-7}$, that means roughly $\log(10^7)/\log(1/0.85) \approx 100$ iterations max.

```python
# Cell 13: The HARP algorithm itself
def harp_rank(
    task_idx,
    beta=(BETA_P, BETA_C, BETA_PHI),
    alpha=ALPHA,
    kappa=KAPPA,
    max_iter=MAX_ITER,
    tol=TOLERANCE,
    return_diagnostics=False,
):
    """
    Run HARP for a single task and return per-agent scores.

    Returns:
        agent_scores: array of length n_agents
        n_iters:      number of iterations until convergence
        residuals:    list of L1 residuals per iteration (if return_diagnostics)
    """
    # Build the three transition matrices for this specific task
    M_P, M_C, M_phi, A_idx, _, _ = make_transition_matrices(task_idx)

    # Convex combination: stays column-stochastic because each M is
    # column-stochastic and the betas sum to 1
    M = beta[0] * M_P + beta[1] * M_C + beta[2] * M_phi

    # Semantic teleport vector
    p = teleport_vector(task_idx, kappa=kappa)

    # Power iteration
    r = p.copy()
    residuals = []
    for it in range(max_iter):
        r_new = alpha * (M @ r) + (1 - alpha) * p
        diff = float(np.linalg.norm(r_new - r, ord=1))
        residuals.append(diff)
        r = r_new
        if diff < tol:
            break

    agent_scores = r[A_idx]
    if return_diagnostics:
        return agent_scores, it + 1, residuals
    return agent_scores, it + 1

# Try it
scores, n_iters = harp_rank(test_idx[0])
top_3 = np.argsort(-scores)[:3]
print(f"Task: \"{TASKS.iloc[test_idx[0]]['desc'][:70]}...\"")
print(f"Converged in {n_iters} iterations.")
print("HARP top 3 agents:")
for rank, ai in enumerate(top_3, 1):
    print(f"  #{rank}: {AGENT_NAMES[ai]:35s}  score={scores[ai]:.4f}")
```

Compare this output to the Sim-only top-3 from Section 12. If they are identical, HARP is being dominated by the semantic signal — try lowering `BETA_PHI`. If they are radically different, HARP is pulling on the performance and endorsement signals heavily — examine whether that reranking is justified by looking at the agents' history.

---

## Section 14 — Defining the baselines

To know whether HARP is doing anything useful, we compare it against five reasonable alternatives. Each baseline ignores some of the signals HARP uses; together they reveal how much each component is worth.

```python
# Cell 14: Baselines for comparison
def baseline_random(task_idx):
    """Pure noise — establishes a floor."""
    return np.random.rand(n_agents)

def baseline_perf_only(task_idx):
    """Global average performance across all skills — ignores the task."""
    return W_P.mean(axis=1)

def baseline_sim_only(task_idx):
    """Cosine similarity between agent description and task — ignores history."""
    return cosine_similarity(agent_emb, task_emb[task_idx:task_idx+1]).ravel()

def baseline_vanilla_pagerank(task_idx):
    """Classical PageRank on the endorsement graph only, uniform teleport."""
    M = col_normalize(W_C)
    G = nx.from_numpy_array(M.T, create_using=nx.DiGraph)
    pr = nx.pagerank(G, alpha=0.85)
    return np.array([pr.get(i, 1.0 / n_agents) for i in range(n_agents)])

def baseline_weighted_avg(task_idx, w=(0.4, 0.3, 0.3)):
    """Naive weighted average of perf + sim + pagerank, after min-max normalization."""
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)
    return (
        w[0] * normalize(baseline_perf_only(task_idx))
        + w[1] * normalize(baseline_sim_only(task_idx))
        + w[2] * normalize(baseline_vanilla_pagerank(task_idx))
    )

# Wrap HARP to match the same calling convention
def baseline_harp(task_idx):
    return harp_rank(task_idx)[0]

print("All baselines defined.")
```

Notice that `baseline_weighted_avg` is the closest competitor: it uses the same three signals as HARP, just combined linearly without the graph structure. The gap between HARP and this baseline is the cleanest measure of "does the random-walk fusion actually help, beyond just averaging?"

---

## Section 15 — Evaluation metrics

We need to score the rankings. Five standard information-retrieval metrics serve different purposes. NDCG@5 captures the overall ordering quality among the top 5. MRR (mean reciprocal rank) is sensitive only to where the first good agent lands. Precision@3 is the most practical: "of the top 3, how many are actually good?" Recall@5 measures coverage. Regret measures how much utility we lose by picking the top-ranked agent vs the truly best one.

```python
# Cell 15: Evaluation metrics
def ndcg_at_k(scores, gains, k=5):
    """Normalized Discounted Cumulative Gain."""
    order = np.argsort(-scores)[:k]
    dcg = sum(gains[order[i]] / np.log2(i + 2) for i in range(k))
    ideal_order = np.argsort(-gains)[:k]
    idcg = sum(gains[ideal_order[i]] / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0

def mean_reciprocal_rank(scores, gains):
    """1 / rank of first relevant agent."""
    for i, ai in enumerate(np.argsort(-scores)):
        if gains[ai] > 0:
            return 1.0 / (i + 1)
    return 0.0

def precision_at_k(scores, gains, k=3):
    """Fraction of top-k that are relevant."""
    return float(np.mean([gains[ai] > 0 for ai in np.argsort(-scores)[:k]]))

def recall_at_k(scores, gains, k=5):
    """Fraction of all relevant agents that appear in top-k."""
    total_pos = max(int((gains > 0).sum()), 1)
    return float(np.sum([gains[ai] > 0 for ai in np.argsort(-scores)[:k]])) / total_pos

def regret(scores, gains):
    """How much utility we miss by picking the top-1 vs the actual best."""
    chosen = int(np.argmax(scores))
    return float((gains.max() - gains[chosen]) / (gains.max() + 1e-9))

def task_gains(task_idx):
    """Ground truth: agents whose mean true skill on this task's required
    skills is above the median are considered 'relevant' (binary gain)."""
    skills_i = task_required_skills[task_idx]
    raw = true_skill[:, skills_i].mean(axis=1)
    return (raw > np.median(raw)).astype(float)

print("Evaluation metrics ready.")
```

The choice to make gains binary (above vs below median) is a common simplification. You could also use graded gains (raw skill values) and get a slightly different NDCG, but binary is easier to interpret as "good agent or not".

---

## Section 16 — Running the full benchmark

Now we loop over every test task, run each method, score each ranking, and aggregate. This is the table you would put in a paper.

```python
# Cell 16: Main benchmark
methods = {
    "Random":      baseline_random,
    "Perf-only":   baseline_perf_only,
    "Sim-only":    baseline_sim_only,
    "VanillaPR":   baseline_vanilla_pagerank,
    "WeightedAvg": baseline_weighted_avg,
    "HARP":        baseline_harp,
}

metric_fns = {
    "NDCG@5": lambda s, g: ndcg_at_k(s, g, k=5),
    "MRR":    mean_reciprocal_rank,
    "P@3":    lambda s, g: precision_at_k(s, g, k=3),
    "R@5":    lambda s, g: recall_at_k(s, g, k=5),
    "Regret": regret,
}

results = {m: {k: [] for k in metric_fns} for m in methods}
for ti in test_idx:
    g = task_gains(ti)
    for method_name, method_fn in methods.items():
        scores = method_fn(ti)
        for metric_name, metric_fn in metric_fns.items():
            results[method_name][metric_name].append(metric_fn(scores, g))

# Build the summary table
summary = pd.DataFrame({
    m: {k: np.mean(v) for k, v in d.items()}
    for m, d in results.items()
}).T

print("=" * 65)
print("MAIN BENCHMARK — Mean metric values across test tasks")
print("=" * 65)
print(summary.round(3))
print()
print("Higher is better for NDCG, MRR, P@3, R@5.")
print("Lower is better for Regret.")
```

You should see HARP at or near the top of every metric except Regret (where it should be near the bottom). If it is not winning, the most common culprits are (a) too few training tasks, so the performance signal is too noisy, or (b) a beta weighting that overcounts a poor-quality signal. Try tuning $(\beta_P, \beta_C, \beta_\phi)$ on a held-out slice of the training tasks.

---

## Section 17 — Ablation study

A benchmark table tells you HARP wins. An ablation tells you *why* HARP wins. We turn off each signal in turn (set its beta to zero, redistribute mass to the others) and see how much NDCG@5 drops.

```python
# Cell 17: Ablation study
ablation_configs = [
    ("HARP-full",              (BETA_P, BETA_C, BETA_PHI),  "softmax"),
    ("HARP-no-Performance",    (0.0,    0.5,    0.5),       "softmax"),
    ("HARP-no-Endorsement",    (0.6,    0.0,    0.4),       "softmax"),
    ("HARP-no-Semantic",       (0.6,    0.4,    0.0),       "softmax"),
    ("HARP-uniform-teleport",  (BETA_P, BETA_C, BETA_PHI),  "uniform"),
]

def harp_with_uniform_teleport(task_idx, beta):
    """HARP variant that uses a uniform teleport instead of semantic."""
    M_P, M_C, M_phi, A_idx, *_ = make_transition_matrices(task_idx)
    M = beta[0] * M_P + beta[1] * M_C + beta[2] * M_phi
    V = M.shape[0]
    p = np.ones(V) / V
    r = p.copy()
    for _ in range(MAX_ITER):
        r_new = ALPHA * (M @ r) + (1 - ALPHA) * p
        if np.linalg.norm(r_new - r, 1) < TOLERANCE:
            break
        r = r_new
    return r[A_idx]

ablation_rows = []
for name, beta, teleport_kind in ablation_configs:
    ndcgs = []
    for ti in test_idx:
        g = task_gains(ti)
        if teleport_kind == "uniform":
            scores = harp_with_uniform_teleport(ti, beta)
        else:
            scores, _ = harp_rank(ti, beta=beta)
        ndcgs.append(ndcg_at_k(scores, g, k=5))
    ablation_rows.append({"Variant": name, "NDCG@5": float(np.mean(ndcgs))})

ablation_df = pd.DataFrame(ablation_rows)
print("=" * 50)
print("ABLATION — Which signal carries the load?")
print("=" * 50)
print(ablation_df.round(3))
print()
print("The drop from HARP-full to each ablation tells you")
print("how much that component contributes.")
```

A healthy ablation should show that *every* component matters — that is, every "HARP-no-X" variant is meaningfully worse than HARP-full. If one component can be removed with no loss, you have a redundant signal and could simplify the system.

---

## Section 18 — Visualizing convergence

Theorem 1 promises that HARP converges geometrically with rate $\alpha$. Let's confirm that empirically by plotting the $\ell_1$ residual at each iteration against the theoretical $\alpha^t$ bound. The two curves should track closely on a log scale.

```python
# Cell 18: Convergence plot
ti = test_idx[0]
_, n_iters, residuals = harp_rank(ti, return_diagnostics=True)

plt.figure(figsize=(8, 4.5))
plt.semilogy(residuals, "o-", label="HARP residual $\\|r_{t+1} - r_t\\|_1$",
             markersize=4)
theoretical = [residuals[0] * (ALPHA ** i) for i in range(len(residuals))]
plt.semilogy(theoretical, "--", label=f"Theory: $\\alpha^t$ bound (α={ALPHA})")
plt.axhline(TOLERANCE, color="red", linestyle=":", label=f"Tolerance {TOLERANCE}")
plt.xlabel("Power iteration step")
plt.ylabel("L1 residual (log scale)")
plt.title("Empirical contraction matches the α-rate guarantee (Theorem 1)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"Converged in {n_iters} iterations.")
```

This plot is doing the mathematical proof in front of your eyes. The straight dashed line is the theoretical worst case; the data points are the actual behavior; the gap between them is how much you could in principle tighten the analysis for this specific instance.

---

## Section 19 — Visualizing the agent score landscape

A heatmap of HARP scores across all test tasks and all agents is the single most informative diagnostic. Each row is a task, each column is an agent, and the cell color is the HARP score. Hot spots show where each agent gets used.

```python
# Cell 19: Heatmap of HARP scores
score_matrix = np.zeros((len(test_idx), n_agents))
for row_i, ti in enumerate(test_idx):
    score_matrix[row_i], _ = harp_rank(ti)

plt.figure(figsize=(12, 6))
sns.heatmap(
    score_matrix,
    xticklabels=AGENT_NAMES,
    yticklabels=[f"t{ti:02d} L{TASKS.iloc[ti].level}" for ti in test_idx],
    cmap="viridis",
    cbar_kws={"label": "HARP score"},
)
plt.xticks(rotation=45, ha="right")
plt.title("HARP agent scores across test tasks")
plt.xlabel("Agent")
plt.ylabel("Task (id + level)")
plt.tight_layout()
plt.show()
```

Look for vertical stripes (some agents always win regardless of task — possibly a sign their generalist description dominates) and horizontal stripes (some tasks have a clear single winner — likely tasks with very specific skill requirements). A healthy heatmap has a mix of both.

---

## Section 20 — Interactive demo: ranking agents for your own task

Finally, the part that makes this an *application*. Type any task description, and HARP will rank all twelve agents for it. This is what you would wire into a production router.

```python
# Cell 20: Interactive ranker for arbitrary new tasks
def rank_agents_for_new_task(task_description, top_k=5):
    """
    Given a free-text task description, return the top-k agents per HARP.
    This is the function you would expose as your routing API endpoint.
    """
    # Embed the new task
    new_emb = embedder.encode([task_description], normalize_embeddings=True)

    # Temporarily extend the task embedding matrix
    global task_emb
    saved_task_emb = task_emb
    task_emb = np.vstack([saved_task_emb, new_emb])
    new_idx = len(task_emb) - 1

    # Score it
    scores, n_iters = harp_rank(new_idx)

    # Restore the original task embeddings
    task_emb = saved_task_emb

    # Format the output
    order = np.argsort(-scores)
    print(f"\nTask: \"{task_description}\"")
    print(f"Converged in {n_iters} iterations. Top {top_k} agents:")
    print("-" * 70)
    for rank, ai in enumerate(order[:top_k], 1):
        print(f"  #{rank}: {AGENT_NAMES[ai]:35s}  score={scores[ai]:.4f}")
    return [(AGENT_NAMES[ai], float(scores[ai])) for ai in order[:top_k]]

# Try it on a few example tasks
example_tasks = [
    "Write a python script that scrapes the top stories from Hacker News and saves them to a CSV.",
    "Read this academic PDF and summarize the methodology section in three bullet points.",
    "Open the attached image of a receipt and extract the total amount and merchant name.",
    "Listen to this podcast clip and transcribe the first two minutes.",
]
for t in example_tasks:
    rank_agents_for_new_task(t, top_k=3)
```

You should see, qualitatively, that the python-scraping task ranks `AutoGen-Coder` or `SWE-Agent-Claude` near the top, the PDF task ranks the deep-research agents highly, the image task favors agents whose description mentions vision, and the audio task ranks specialist transcribers (if any) ahead of pure coders. If you see weird mismatches, that is informative — it usually means either the agent description is too vague or the history is too sparse for that skill category.

---

## What you have built

You now have a complete agent ranker that fuses performance history, endorsement structure, and semantic similarity into a single principled score, with provable convergence and a clean way to handle cold-start agents. It runs in under thirty seconds end-to-end on Colab's free tier. The most natural extensions, listed roughly in order of effort, are wiring it to a real GAIA dataset by replacing Section 5 with `load_dataset`, adding a Thompson-sampling exploration layer on top for online deployment, swapping the sentence transformer for a stronger reranker like BGE-M3, and time-decaying the history so recent outcomes dominate stale ones.

If you want to push deeper into research territory, the discussion section of the writeup sketches federated trust aggregation, adversarial-robust seed sets, and the tensor extension that gives up uniqueness for higher-order Markov structure. Any of those is a real paper waiting to be written.
