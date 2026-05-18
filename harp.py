"""
HARP — Hybrid Agent Ranking via Personalized PageRank
======================================================

A complete, self-contained reference implementation of HARP, a multi-relational
personalized-PageRank algorithm for ranking LLM agents against incoming tasks.

HARP fuses three signals into a single principled score per agent:
    1. Performance history (Bayesian-smoothed empirical success rates)
    2. Endorsement / collaboration graph among agents (EigenTrust-style)
    3. Skill-task semantic similarity from sentence embeddings

The algorithm constructs three column-stochastic transition matrices, combines
them convexly into a single Markov operator M_tau conditioned on the current
task tau, and runs power iteration with a softmax-temperature semantic
teleport vector p_tau:

    r_{t+1} = alpha * M_tau * r_t + (1 - alpha) * p_tau

This is provably a contraction in L1 with rate alpha, so convergence to a
unique stationary distribution is guaranteed by the Banach fixed-point theorem
(see the accompanying research writeup for proofs).

USAGE
-----
# In Colab, paste into a cell and run:
!pip -q install sentence-transformers networkx scikit-learn seaborn
%run harp.py

# From the terminal:
pip install sentence-transformers networkx scikit-learn seaborn matplotlib pandas
python harp.py

# To skip the demo and import functions:
from harp import harp_rank, rank_agents_for_new_task

REQUIREMENTS
------------
python >= 3.9
numpy, pandas, scikit-learn, networkx, sentence-transformers,
matplotlib, seaborn
"""

import argparse
import random
import sys
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAVE_SNS = True
except ImportError:
    HAVE_SNS = False

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =============================================================================
# Global configuration
# =============================================================================
# These hyperparameters can be tuned per deployment. The defaults are the
# values recommended in the HARP writeup based on synthetic-GAIA experiments.
ALPHA = 0.85         # Damping factor (same default as Brin & Page 1998)
KAPPA = 10.0         # Temperature for the softmax teleport vector
BETA_P = 0.5         # Weight on performance transition matrix M^P
BETA_C = 0.3         # Weight on endorsement transition matrix M^C
BETA_PHI = 0.2       # Weight on semantic transition matrix M^phi
TOLERANCE = 1e-7     # Convergence threshold for power iteration (L1 norm)
MAX_ITER = 200       # Safety cap on power iterations

# Seeds for reproducibility
SEED = 42


# =============================================================================
# The agent universe
# =============================================================================
# In a production deployment you would load this from your agent registry.
# Each agent gets a free-text description that the semantic similarity signal
# will compare against incoming task descriptions.
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

AGENT_DESC = {
    "HAL-Generalist-Sonnet-4.5":
        "general purpose assistant with web browsing, code execution, "
        "and image understanding tools",
    "HAL-Generalist-GPT-5":
        "general purpose assistant with strong reasoning, code, "
        "and multi-step planning",
    "HAL-Generalist-Gemini-2.5":
        "general purpose assistant with multimodal vision and long-context "
        "reading",
    "HF-OpenDeepResearch-GPT-4o":
        "deep research agent that browses the web and reads PDF documents "
        "to answer questions",
    "HF-OpenDeepResearch-Llama3-70B":
        "open source deep research agent for web search and document reading",
    "AutoGen-Coder":
        "python code generation agent with iterative execution and debugging",
    "MetaGPT-Engineer":
        "software engineering multi-agent pipeline for writing and testing "
        "code",
    "CAMEL-Researcher":
        "role-playing research assistant for structured analysis and writing",
    "Voyager-Skill-Agent":
        "lifelong skill library agent that composes tools from past "
        "experience",
    "GPTSwarm-Swarm":
        "optimizable agent graph that combines several specialists in "
        "parallel",
    "WebArena-Browser":
        "specialist web browsing agent that navigates websites and clicks "
        "UI elements",
    "SWE-Agent-Claude":
        "software engineering agent for editing code repositories and "
        "writing patches",
}


# =============================================================================
# The skill vocabulary
# =============================================================================
# Skills are the bridge between agents and tasks. GAIA's annotator metadata
# uses a free-form Tools field; we discretize to eight categories that cover
# what GAIA L1/L2/L3 tasks demand.
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
        "navigating interactive websites, clicking buttons, filling forms, "
        "and reading rendered pages",
    "pdf_reading":
        "extracting and understanding text and tables from PDF documents",
    "python_coding":
        "writing, executing, and debugging python programs to solve "
        "computational tasks",
    "image_recognition":
        "understanding the content of images, photographs, diagrams, and "
        "screenshots",
    "math_calculation":
        "performing arithmetic, algebra, and symbolic mathematics with "
        "precision",
    "audio_transcription":
        "transcribing spoken language from audio files into text",
    "spreadsheet_analysis":
        "reading excel and csv files and computing aggregations over rows "
        "and columns",
    "web_search":
        "issuing search engine queries and synthesizing information from "
        "results",
}


# =============================================================================
# Task generation (GAIA-style synthetic tasks)
# =============================================================================
# In production you would stream tasks from a queue. For the demo we generate
# synthetic tasks with GAIA's level distribution (31% L1, 53% L2, 16% L3).
TASK_TEMPLATES = [
    "Find the population of {place} from the official census PDF and compute "
    "its growth rate.",
    "Open this spreadsheet and return the sum of column B for entries in 2024.",
    "Transcribe the attached audio clip and identify the speaker.",
    "Browse the wikipedia page for {place} and extract the founding year.",
    "Run the attached python script and report the final output.",
    "Identify the country shown in this satellite image and look up its "
    "capital city.",
    "Search the web for the latest CEO of {place} and find their previous "
    "role.",
    "Open the attached PDF and find the date of the third figure caption.",
    "Compute the integral of the expression on page 4 of the document.",
    "Listen to the recording and write a five sentence summary in english.",
]

PLACE_FILLERS = [
    "Paris", "Brazil", "Tokyo", "Apollo 11", "Marie Curie",
    "the Eiffel Tower", "OpenAI", "Anthropic", "Mumbai", "Iceland",
]


def generate_tasks(n_tasks: int = 30, seed: int = SEED) -> pd.DataFrame:
    """
    Generate synthetic GAIA-style tasks with a level distribution matching
    the actual GAIA validation split.

    The level field is preserved for downstream stratified analysis (e.g.
    measuring whether HARP gains more on L3 tasks than L1).
    """
    rng = np.random.default_rng(seed)
    levels = rng.choice([1, 2, 3], size=n_tasks, p=[0.31, 0.53, 0.16])
    rows = []
    for i, lvl in enumerate(levels):
        template = TASK_TEMPLATES[rng.integers(0, len(TASK_TEMPLATES))]
        place = PLACE_FILLERS[rng.integers(0, len(PLACE_FILLERS))]
        desc = template.format(place=place) if "{place}" in template else template
        rows.append({
            "task_id": f"task_{i:03d}",
            "desc": desc,
            "level": int(lvl),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Outcome history simulation
# =============================================================================
def simulate_history(
    agent_emb: np.ndarray,
    skill_emb: np.ndarray,
    task_emb: np.ndarray,
    train_idx: List[int],
    seed: int = SEED,
) -> Tuple[pd.DataFrame, np.ndarray, List[List[int]]]:
    """
    Simulate a sparse outcome history.

    Each agent has a latent "true skill" vector drawn from a Beta distribution.
    Each task implicitly requires the two skills whose embeddings are most
    similar to it. For each training task, 4-6 random agents attempt it; each
    succeeds with probability equal to its mean true skill on the required
    skills.

    Returns:
        H:                    DataFrame with (agent, task, skill, outcome) rows
        true_skill:           (n_agents, n_skills) ground-truth skill matrix
        task_required_skills: list of [skill_idx, skill_idx] per task
    """
    rng = np.random.default_rng(seed)
    n_agents = len(AGENT_NAMES)
    n_skills = len(SKILLS)

    # Latent ground-truth skill matrix
    true_skill = np.clip(
        rng.beta(2, 2, size=(n_agents, n_skills))
        + 0.15 * rng.standard_normal(size=(n_agents, n_skills)),
        0.02, 0.98,
    )

    # Each task's required skills = top-2 most similar by embedding
    task_skill_sim = cosine_similarity(task_emb, skill_emb)
    task_required_skills = [np.argsort(-row)[:2].tolist() for row in task_skill_sim]

    # Simulate attempts
    history_rows = []
    for ti in train_idx:
        skills_i = task_required_skills[ti]
        n_attempters = rng.integers(4, 7)
        attempters = rng.choice(n_agents, size=n_attempters, replace=False)
        for ai in attempters:
            p_success = float(np.mean(true_skill[ai, skills_i]))
            outcome = int(rng.random() < p_success)
            for si in skills_i:
                history_rows.append({
                    "agent": int(ai),
                    "task": int(ti),
                    "skill": int(si),
                    "outcome": outcome,
                })

    H = pd.DataFrame(history_rows)
    return H, true_skill, task_required_skills


# =============================================================================
# Matrix construction helpers
# =============================================================================
def beta_smoothed(successes: int, attempts: int) -> float:
    """
    Beta(1,1)-smoothed success rate. Returns the posterior mean of the
    underlying success probability given (successes, attempts).
    With zero attempts this returns 0.5 — the cold-start prior.
    """
    failures = attempts - successes
    return (1 + successes) / (2 + successes + failures)


def col_normalize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize columns of W to sum to 1. All-zero columns are left untouched
    (the caller applies the uniform dangling-node fix afterwards).
    """
    col_sums = W.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums < eps, 1.0, col_sums)
    return W / col_sums


def build_performance_matrix(H: pd.DataFrame) -> np.ndarray:
    """
    Build the agent-skill performance weight matrix W^P with Beta(1,1)
    smoothing. Shape: (n_agents, n_skills).
    """
    n_agents = len(AGENT_NAMES)
    n_skills = len(SKILLS)
    W_P = np.zeros((n_agents, n_skills))
    for ai in range(n_agents):
        for si in range(n_skills):
            subset = H[(H.agent == ai) & (H.skill == si)]
            n_succ = int(subset.outcome.sum())
            n_att = len(subset)
            W_P[ai, si] = beta_smoothed(n_succ, n_att)
    return W_P


def build_endorsement_matrix(H: pd.DataFrame, train_idx: List[int]) -> np.ndarray:
    """
    Build the agent-agent endorsement matrix W^C via co-success / co-failure.
    Shape: (n_agents, n_agents).

    Reward: if agents i and j both succeed on the same task, w[i,j] += 1.
    Penalty: if agents i and j both fail on the same task, w[i,j] -= 0.3
    (clipped at zero).
    """
    n_agents = len(AGENT_NAMES)
    W_C = np.zeros((n_agents, n_agents))
    for ti in train_idx:
        succ = H[(H.task == ti) & (H.outcome == 1)].agent.unique()
        for i in succ:
            for j in succ:
                if i != j:
                    W_C[i, j] += 1.0
    for ti in train_idx:
        fail = H[(H.task == ti) & (H.outcome == 0)].agent.unique()
        for i in fail:
            for j in fail:
                if i != j:
                    W_C[i, j] = max(W_C[i, j] - 0.3, 0.0)
    return W_C


def make_transition_matrices(
    task_idx: int,
    W_P: np.ndarray,
    W_C: np.ndarray,
    agent_emb: np.ndarray,
    skill_emb: np.ndarray,
    task_emb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build the three task-conditional column-stochastic transition matrices
    over the state space V = agents ∪ skills ∪ {task}.

    M^P:   Agent <-> Skill via performance
    M^C:   Agent <-> Agent via endorsement
    M^phi: Skill <-> Task and Agent <-> Task via semantic similarity

    Returns (M_P, M_C, M_phi, A_idx, S_idx, T_idx) where the last three are
    the row/column index ranges of each node group.
    """
    n_agents = len(AGENT_NAMES)
    n_skills = len(SKILLS)
    V = n_agents + n_skills + 1
    A_idx = np.arange(0, n_agents)
    S_idx = np.arange(n_agents, n_agents + n_skills)
    T_idx = n_agents + n_skills

    # ---------- M^P: Agent <-> Skill ----------
    M_P = np.zeros((V, V))
    P_skill_to_agent = col_normalize(W_P.T)
    P_agent_to_skill = col_normalize(W_P)
    for ai in range(n_agents):
        for si in range(n_skills):
            M_P[A_idx[ai], S_idx[si]] = P_skill_to_agent[si, ai]
            M_P[S_idx[si], A_idx[ai]] = P_agent_to_skill[ai, si]
    M_P[T_idx, T_idx] = 1.0
    zero_cols = M_P.sum(axis=0) < 1e-12
    M_P[:, zero_cols] = 1.0 / V

    # ---------- M^C: Agent <-> Agent ----------
    M_C = np.eye(V)
    C_agent = col_normalize(W_C)
    for i in range(n_agents):
        M_C[A_idx[i], A_idx] = 0.0
        M_C[A_idx, A_idx[i]] = C_agent[:, i]
    zero_cols = M_C.sum(axis=0) < 1e-12
    M_C[:, zero_cols] = 1.0 / V

    # ---------- M^phi: Skill/Agent <-> Task semantic ----------
    M_phi = np.zeros((V, V))
    sim_skill_task = (1 + cosine_similarity(
        skill_emb, task_emb[task_idx:task_idx + 1]).ravel()) / 2
    sim_agent_task = (1 + cosine_similarity(
        agent_emb, task_emb[task_idx:task_idx + 1]).ravel()) / 2
    PHI_to_skills = sim_skill_task / max(sim_skill_task.sum(), 1e-12)
    PHI_to_agents = sim_agent_task / max(sim_agent_task.sum(), 1e-12)
    M_phi[S_idx, T_idx] = 0.5 * PHI_to_skills
    M_phi[A_idx, T_idx] = 0.5 * PHI_to_agents
    M_phi[T_idx, S_idx] = sim_skill_task / (sim_skill_task.sum() + 1e-12)
    zero_cols = M_phi.sum(axis=0) < 1e-12
    M_phi[:, zero_cols] = 1.0 / V

    return M_P, M_C, M_phi, A_idx, S_idx, T_idx


def teleport_vector(
    task_idx: int,
    agent_emb: np.ndarray,
    skill_emb: np.ndarray,
    task_emb: np.ndarray,
    kappa: float = KAPPA,
) -> np.ndarray:
    """
    Softmax-temperature semantic teleport vector p_tau over V.

    Mass concentrates on agents and skills whose descriptions are most
    semantically similar to the current task. The task node itself receives
    zero mass.
    """
    n_agents = len(AGENT_NAMES)
    n_skills = len(SKILLS)
    V = n_agents + n_skills + 1
    sim_agents = cosine_similarity(
        agent_emb, task_emb[task_idx:task_idx + 1]).ravel()
    sim_skills = cosine_similarity(
        skill_emb, task_emb[task_idx:task_idx + 1]).ravel()
    sims = np.concatenate([sim_agents, sim_skills, np.array([0.0])])
    p = np.exp(kappa * sims)
    p = p / p.sum()
    return p


# =============================================================================
# The HARP algorithm
# =============================================================================
def harp_rank(
    task_idx: int,
    W_P: np.ndarray,
    W_C: np.ndarray,
    agent_emb: np.ndarray,
    skill_emb: np.ndarray,
    task_emb: np.ndarray,
    beta: Tuple[float, float, float] = (BETA_P, BETA_C, BETA_PHI),
    alpha: float = ALPHA,
    kappa: float = KAPPA,
    max_iter: int = MAX_ITER,
    tol: float = TOLERANCE,
    return_diagnostics: bool = False,
):
    """
    Run HARP for a single task and return per-agent scores.

    The update rule is:
        r_{t+1} = alpha * M_tau * r_t + (1 - alpha) * p_tau
    where:
        M_tau = beta[0] * M^P + beta[1] * M^C + beta[2] * M^phi
        p_tau = softmax(kappa * cos(phi(task), phi(node)))

    Returns:
        agent_scores (n_agents,), n_iters, [residuals if return_diagnostics]
    """
    M_P, M_C, M_phi, A_idx, _, _ = make_transition_matrices(
        task_idx, W_P, W_C, agent_emb, skill_emb, task_emb,
    )
    # Convex combination — column-stochastic because each summand is
    # column-stochastic and the betas sum to 1.
    M = beta[0] * M_P + beta[1] * M_C + beta[2] * M_phi

    p = teleport_vector(task_idx, agent_emb, skill_emb, task_emb, kappa=kappa)

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


# =============================================================================
# Baselines for comparison
# =============================================================================
def baseline_random(task_idx, **kwargs):
    return np.random.default_rng(task_idx).random(len(AGENT_NAMES))


def baseline_perf_only(task_idx, *, W_P, **kwargs):
    return W_P.mean(axis=1)


def baseline_sim_only(task_idx, *, agent_emb, task_emb, **kwargs):
    return cosine_similarity(agent_emb, task_emb[task_idx:task_idx + 1]).ravel()


def baseline_vanilla_pagerank(task_idx, *, W_C, **kwargs):
    n_agents = len(AGENT_NAMES)
    M = col_normalize(W_C)
    G = nx.from_numpy_array(M.T, create_using=nx.DiGraph)
    pr = nx.pagerank(G, alpha=0.85)
    return np.array([pr.get(i, 1.0 / n_agents) for i in range(n_agents)])


def baseline_weighted_avg(task_idx, *, W_P, W_C, agent_emb, task_emb,
                          w=(0.4, 0.3, 0.3), **kwargs):
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)
    a = baseline_perf_only(task_idx, W_P=W_P)
    b = baseline_sim_only(task_idx, agent_emb=agent_emb, task_emb=task_emb)
    c = baseline_vanilla_pagerank(task_idx, W_C=W_C)
    return w[0] * normalize(a) + w[1] * normalize(b) + w[2] * normalize(c)


# =============================================================================
# Evaluation metrics
# =============================================================================
def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    order = np.argsort(-scores)[:k]
    dcg = sum(gains[order[i]] / np.log2(i + 2) for i in range(k))
    ideal_order = np.argsort(-gains)[:k]
    idcg = sum(gains[ideal_order[i]] / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(scores: np.ndarray, gains: np.ndarray) -> float:
    """1 / rank of first relevant agent."""
    for i, ai in enumerate(np.argsort(-scores)):
        if gains[ai] > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 3) -> float:
    """Fraction of top-k that are relevant."""
    return float(np.mean([gains[ai] > 0 for ai in np.argsort(-scores)[:k]]))


def recall_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 5) -> float:
    """Fraction of all relevant agents that appear in top-k."""
    total_pos = max(int((gains > 0).sum()), 1)
    return float(np.sum([gains[ai] > 0 for ai in np.argsort(-scores)[:k]])) / total_pos


def regret(scores: np.ndarray, gains: np.ndarray) -> float:
    """Utility missed by picking top-1 vs the actual best (lower = better)."""
    chosen = int(np.argmax(scores))
    return float((gains.max() - gains[chosen]) / (gains.max() + 1e-9))


def task_gains(task_idx: int, true_skill: np.ndarray,
               task_required_skills: List[List[int]]) -> np.ndarray:
    """Binary relevance: agents above the median true-skill on this task."""
    skills_i = task_required_skills[task_idx]
    raw = true_skill[:, skills_i].mean(axis=1)
    return (raw > np.median(raw)).astype(float)


# =============================================================================
# Convenience: rank agents for a brand-new task description
# =============================================================================
def rank_agents_for_new_task(
    task_description: str,
    embedder: SentenceTransformer,
    W_P: np.ndarray,
    W_C: np.ndarray,
    agent_emb: np.ndarray,
    skill_emb: np.ndarray,
    task_emb_base: np.ndarray,
    top_k: int = 5,
    verbose: bool = True,
) -> List[Tuple[str, float]]:
    """
    Given a free-text task description, return the top-k agents per HARP.
    This is the function you would expose as your routing API endpoint.
    """
    new_emb = embedder.encode([task_description], normalize_embeddings=True)
    # Build a temporary task embedding matrix with the new task appended.
    task_emb = np.vstack([task_emb_base, new_emb])
    new_idx = len(task_emb) - 1

    scores, n_iters = harp_rank(
        new_idx, W_P, W_C, agent_emb, skill_emb, task_emb,
    )
    order = np.argsort(-scores)
    if verbose:
        print(f"\nTask: \"{task_description}\"")
        print(f"Converged in {n_iters} iterations. Top {top_k} agents:")
        print("-" * 70)
        for rank, ai in enumerate(order[:top_k], 1):
            print(f"  #{rank}: {AGENT_NAMES[ai]:35s}  score={scores[ai]:.4f}")
    return [(AGENT_NAMES[ai], float(scores[ai])) for ai in order[:top_k]]


# =============================================================================
# End-to-end demo
# =============================================================================
def run_demo(show_plots: bool = True) -> None:
    """
    Run the full HARP pipeline end-to-end: generate data, build matrices,
    run the main benchmark, run the ablation, render diagnostic plots, and
    finish with an interactive demo on four example tasks.
    """
    np.random.seed(SEED)
    random.seed(SEED)

    n_agents = len(AGENT_NAMES)
    n_skills = len(SKILLS)

    print("=" * 70)
    print("HARP — Hybrid Agent Ranking via Personalized PageRank")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  alpha={ALPHA}, kappa={KAPPA}, beta=({BETA_P}, {BETA_C}, {BETA_PHI})")
    print(f"  agents={n_agents}, skills={n_skills}")

    # --- Step 1: Generate tasks ---
    print("\n[1/8] Generating GAIA-style tasks...")
    TASKS = generate_tasks(n_tasks=30, seed=SEED)
    print(f"      {len(TASKS)} tasks: "
          f"L1={sum(TASKS.level==1)}, L2={sum(TASKS.level==2)}, "
          f"L3={sum(TASKS.level==3)}")

    # --- Step 2: Embed everything ---
    print("\n[2/8] Loading sentence transformer (~80MB on first run)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    agent_emb = embedder.encode(
        [AGENT_DESC[a] for a in AGENT_NAMES], normalize_embeddings=True)
    skill_emb = embedder.encode(
        [SKILL_DESC[s] for s in SKILLS], normalize_embeddings=True)
    task_emb = embedder.encode(
        TASKS["desc"].tolist(), normalize_embeddings=True)
    print(f"      Embeddings: agents={agent_emb.shape}, "
          f"skills={skill_emb.shape}, tasks={task_emb.shape}")

    # --- Step 3: Simulate history ---
    print("\n[3/8] Simulating outcome history...")
    n_train = len(TASKS) // 2
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, len(TASKS)))
    H, true_skill, task_required_skills = simulate_history(
        agent_emb, skill_emb, task_emb, train_idx, seed=SEED,
    )
    print(f"      |H|={len(H)} outcomes  "
          f"(train tasks={len(train_idx)}, test={len(test_idx)})  "
          f"success rate={H.outcome.mean():.2%}")

    # --- Step 4: Build matrices ---
    print("\n[4/8] Building performance and endorsement matrices...")
    W_P = build_performance_matrix(H)
    W_C = build_endorsement_matrix(H, train_idx)
    print(f"      W^P shape={W_P.shape} mean={W_P.mean():.3f}")
    print(f"      W^C shape={W_C.shape} density={(W_C>0).mean():.2%}")

    # --- Step 5: Verify column-stochasticity ---
    print("\n[5/8] Verifying column-stochasticity (Theorem 1 prerequisite)...")
    M_P, M_C, M_phi, *_ = make_transition_matrices(
        test_idx[0], W_P, W_C, agent_emb, skill_emb, task_emb,
    )
    assert np.allclose(M_P.sum(axis=0), 1.0)
    assert np.allclose(M_C.sum(axis=0), 1.0)
    assert np.allclose(M_phi.sum(axis=0), 1.0)
    print("      All three M matrices are column-stochastic. ✓")

    # --- Step 6: Main benchmark ---
    print("\n[6/8] Running main benchmark across all test tasks...")

    methods = {
        "Random":      lambda ti: baseline_random(ti),
        "Perf-only":   lambda ti: baseline_perf_only(ti, W_P=W_P),
        "Sim-only":    lambda ti: baseline_sim_only(
                            ti, agent_emb=agent_emb, task_emb=task_emb),
        "VanillaPR":   lambda ti: baseline_vanilla_pagerank(ti, W_C=W_C),
        "WeightedAvg": lambda ti: baseline_weighted_avg(
                            ti, W_P=W_P, W_C=W_C,
                            agent_emb=agent_emb, task_emb=task_emb),
        "HARP":        lambda ti: harp_rank(
                            ti, W_P, W_C, agent_emb, skill_emb, task_emb)[0],
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
        g = task_gains(ti, true_skill, task_required_skills)
        for method_name, method_fn in methods.items():
            scores = method_fn(ti)
            for metric_name, metric_fn in metric_fns.items():
                results[method_name][metric_name].append(metric_fn(scores, g))

    summary = pd.DataFrame({
        m: {k: float(np.mean(v)) for k, v in d.items()}
        for m, d in results.items()
    }).T

    print("\n" + "=" * 70)
    print("MAIN BENCHMARK — Mean metric values across test tasks")
    print("=" * 70)
    print(summary.round(3).to_string())
    print("\n(Higher better for NDCG/MRR/P@3/R@5; lower better for Regret.)")

    # --- Step 7: Ablation ---
    print("\n[7/8] Running ablation study...")

    def harp_uniform(task_idx, beta):
        M_P, M_C, M_phi, A_idx, *_ = make_transition_matrices(
            task_idx, W_P, W_C, agent_emb, skill_emb, task_emb,
        )
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

    ablation_configs = [
        ("HARP-full",             (BETA_P, BETA_C, BETA_PHI), "softmax"),
        ("HARP-no-Performance",   (0.0,    0.5,    0.5),      "softmax"),
        ("HARP-no-Endorsement",   (0.6,    0.0,    0.4),      "softmax"),
        ("HARP-no-Semantic",      (0.6,    0.4,    0.0),      "softmax"),
        ("HARP-uniform-teleport", (BETA_P, BETA_C, BETA_PHI), "uniform"),
    ]

    ablation_rows = []
    for name, beta, kind in ablation_configs:
        ndcgs = []
        for ti in test_idx:
            g = task_gains(ti, true_skill, task_required_skills)
            if kind == "uniform":
                s = harp_uniform(ti, beta)
            else:
                s, _ = harp_rank(ti, W_P, W_C, agent_emb, skill_emb, task_emb,
                                 beta=beta)
            ndcgs.append(ndcg_at_k(s, g, k=5))
        ablation_rows.append({"Variant": name, "NDCG@5": float(np.mean(ndcgs))})

    abl_df = pd.DataFrame(ablation_rows)
    print("\n" + "=" * 50)
    print("ABLATION — Which signal carries the load?")
    print("=" * 50)
    print(abl_df.round(3).to_string(index=False))

    # --- Step 8: Plots ---
    if show_plots:
        print("\n[8/8] Rendering diagnostic plots...")

        # Convergence plot
        _, n_iters, residuals = harp_rank(
            test_idx[0], W_P, W_C, agent_emb, skill_emb, task_emb,
            return_diagnostics=True,
        )
        plt.figure(figsize=(8, 4.5))
        plt.semilogy(residuals, "o-",
                     label=r"HARP residual $\|r_{t+1}-r_t\|_1$",
                     markersize=4)
        theoretical = [residuals[0] * (ALPHA ** i) for i in range(len(residuals))]
        plt.semilogy(theoretical, "--",
                     label=fr"Theory: $\alpha^t$ bound (α={ALPHA})")
        plt.axhline(TOLERANCE, color="red", linestyle=":",
                    label=f"Tolerance {TOLERANCE}")
        plt.xlabel("Power iteration step")
        plt.ylabel("L1 residual (log scale)")
        plt.title("Empirical contraction matches the α-rate guarantee (Theorem 1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("harp_convergence.png", dpi=120)
        plt.close()

        # Score heatmap (only if seaborn available; else skip)
        if HAVE_SNS:
            score_matrix = np.zeros((len(test_idx), n_agents))
            for row_i, ti in enumerate(test_idx):
                score_matrix[row_i], _ = harp_rank(
                    ti, W_P, W_C, agent_emb, skill_emb, task_emb,
                )
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                score_matrix,
                xticklabels=AGENT_NAMES,
                yticklabels=[f"t{ti:02d} L{TASKS.iloc[ti].level}"
                             for ti in test_idx],
                cmap="viridis",
                cbar_kws={"label": "HARP score"},
            )
            plt.xticks(rotation=45, ha="right")
            plt.title("HARP agent scores across test tasks")
            plt.xlabel("Agent")
            plt.ylabel("Task (id + level)")
            plt.tight_layout()
            plt.savefig("harp_heatmap.png", dpi=120)
            plt.close()
            print("      Saved harp_convergence.png and harp_heatmap.png")
        else:
            print("      Saved harp_convergence.png (install seaborn for heatmap)")

    # --- Interactive demo ---
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO — Ranking agents for new task descriptions")
    print("=" * 70)

    example_tasks = [
        "Write a python script that scrapes the top stories from Hacker News "
        "and saves them to a CSV.",
        "Read this academic PDF and summarize the methodology section in "
        "three bullet points.",
        "Open the attached image of a receipt and extract the total amount "
        "and merchant name.",
        "Listen to this podcast clip and transcribe the first two minutes.",
    ]
    for t in example_tasks:
        rank_agents_for_new_task(
            t, embedder, W_P, W_C, agent_emb, skill_emb, task_emb, top_k=3,
        )

    print("\n" + "=" * 70)
    print("Done. To rank a custom task in Python:")
    print("=" * 70)
    print("""
    from harp import (
        rank_agents_for_new_task, generate_tasks, simulate_history,
        build_performance_matrix, build_endorsement_matrix,
        AGENT_NAMES, AGENT_DESC, SKILLS, SKILL_DESC,
    )
    # ... follow run_demo() above for the setup steps ...
    rank_agents_for_new_task("your task here", embedder, W_P, W_C,
                             agent_emb, skill_emb, task_emb, top_k=5)
    """)


# =============================================================================
# Command-line entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HARP: Hybrid Agent Ranking via Personalized PageRank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating the diagnostic plots (faster).",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Rank agents for a single custom task and exit "
             "(skips the benchmark).",
    )
    args = parser.parse_args()

    if args.task:
        # Quick path: just rank one user-supplied task.
        np.random.seed(SEED)
        random.seed(SEED)
        TASKS = generate_tasks(n_tasks=30, seed=SEED)
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        agent_emb = embedder.encode(
            [AGENT_DESC[a] for a in AGENT_NAMES], normalize_embeddings=True)
        skill_emb = embedder.encode(
            [SKILL_DESC[s] for s in SKILLS], normalize_embeddings=True)
        task_emb = embedder.encode(
            TASKS["desc"].tolist(), normalize_embeddings=True)
        n_train = len(TASKS) // 2
        train_idx = list(range(n_train))
        H, _, _ = simulate_history(
            agent_emb, skill_emb, task_emb, train_idx, seed=SEED)
        W_P = build_performance_matrix(H)
        W_C = build_endorsement_matrix(H, train_idx)
        rank_agents_for_new_task(
            args.task, embedder, W_P, W_C, agent_emb, skill_emb, task_emb,
            top_k=5,
        )
    else:
        run_demo(show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
