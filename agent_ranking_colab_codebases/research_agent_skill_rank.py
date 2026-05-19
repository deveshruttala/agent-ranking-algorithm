
"""
AURA-Rank Research Notebook Codebase
====================================

Goal:
Build and test a PageRank-like ranking algorithm for an MCP / AI Agent Hub.
The algorithm ranks:
1. Agents globally by graph authority: AgentRank.
2. Skills for a task: SkillRank.
3. Agents for a specific task: AURA-Rank.
4. Multi-agent chains: ChainRank.

This script is Google Colab friendly. You can paste it into one Colab notebook
or run it as a .py file.

Author: AgentHub MVP research prototype
"""

# ============================================================
# 0. Optional Colab installs
# ============================================================
# In Google Colab, these libraries are usually already installed.
# If needed, uncomment:
# !pip -q install numpy pandas scikit-learn matplotlib

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required. In Colab run: !pip -q install scikit-learn"
    ) from e

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

random.seed(42)
np.random.seed(42)


# ============================================================
# 1. Data model
# ============================================================

@dataclass
class Agent:
    id: str
    name: str
    description: str
    skills: Dict[str, float]  # skill_id -> proficiency 0..1
    permissions: Set[str]
    required_auth: Set[str]
    memory_scopes: Set[str]
    tools: List[str]
    cost_per_1k_tokens: float
    p95_latency_ms: int
    success_rate: float
    user_success_rate: float
    workspace_success_rate: float
    trust_score: float
    risk_level: float  # 0 low, 1 critical
    verified: bool = True
    install_count: int = 0
    rating: float = 4.5


@dataclass
class Skill:
    id: str
    name: str
    description: str
    keywords: List[str]
    historical_demand: float = 0.5


@dataclass
class MemoryItem:
    id: str
    title: str
    content: str
    scope: str
    sensitivity: float  # 0..1
    allowed_agents: Set[str] = field(default_factory=set)
    blocked_agents: Set[str] = field(default_factory=set)
    allowed_clients: Set[str] = field(default_factory=set)
    blocked_clients: Set[str] = field(default_factory=set)
    historical_usefulness: float = 0.5
    authority: float = 0.5


@dataclass
class RankingContext:
    user_id: str
    workspace_id: str
    client_name: str
    client_permissions: Set[str]
    connected_auth: Set[str]
    allowed_memory_scopes: Set[str]
    budget_per_run_usd: float = 0.25
    target_latency_ms: int = 5000


# ============================================================
# 2. Seed sample skill catalog
# ============================================================

SKILLS: Dict[str, Skill] = {
    "web_research": Skill(
        "web_research", "Web Research",
        "Search web, gather factual information, compare sources, summarize findings.",
        ["research", "search", "web", "sources", "competitor", "market"]
    ),
    "gmail_read": Skill(
        "gmail_read", "Gmail Read",
        "Read emails, summarize inbox, extract sender, subject, dates, attachments.",
        ["email", "gmail", "inbox", "unread", "summarize emails"]
    ),
    "gmail_send": Skill(
        "gmail_send", "Gmail Send",
        "Draft and send emails after approval.",
        ["send email", "draft email", "reply", "outreach"]
    ),
    "calendar_read": Skill(
        "calendar_read", "Calendar Read",
        "Read meetings, availability, events, schedules.",
        ["calendar", "schedule", "availability", "meeting"]
    ),
    "calendar_write": Skill(
        "calendar_write", "Calendar Write",
        "Create, update, or delete calendar events.",
        ["add to calendar", "create event", "schedule meeting", "update calendar"]
    ),
    "github_triage": Skill(
        "github_triage", "GitHub Triage",
        "Analyze GitHub issues, pull requests, labels, milestones, and code changes.",
        ["github", "issue", "pull request", "bug", "repo", "triage", "code"]
    ),
    "stripe_analytics": Skill(
        "stripe_analytics", "Stripe Analytics",
        "Analyze Stripe payments, revenue, refunds, MRR, ARR, and customer activity.",
        ["stripe", "revenue", "payment", "mrr", "arr", "refund", "customer"]
    ),
    "travel_search": Skill(
        "travel_search", "Travel Search",
        "Search flights, hotels, routes, travel options, itineraries, and prices.",
        ["travel", "flight", "hotel", "trip", "itinerary", "booking", "airport"]
    ),
    "travel_book": Skill(
        "travel_book", "Travel Booking",
        "Book travel tickets, hotels, and reservations after explicit approval.",
        ["book flight", "book hotel", "reservation", "ticket"]
    ),
    "writing": Skill(
        "writing", "Writing",
        "Write reports, summaries, emails, briefs, documentation, and articles.",
        ["write", "draft", "report", "summary", "brief", "document"]
    ),
    "qa_review": Skill(
        "qa_review", "QA Review",
        "Review outputs, detect errors, validate claims, test plans, and check quality.",
        ["qa", "review", "validate", "test", "check", "quality"]
    ),
    "memory_retrieval": Skill(
        "memory_retrieval", "Memory Retrieval",
        "Retrieve scoped long-term memory and personalize answers.",
        ["memory", "preference", "context", "remember"]
    ),
}


# ============================================================
# 3. Seed sample agents
# ============================================================

AGENTS: List[Agent] = [
    Agent(
        id="research_agent",
        name="Personal Research Agent",
        description="Finds reliable information, compares sources, and summarizes research.",
        skills={"web_research": 0.95, "writing": 0.72, "memory_retrieval": 0.70},
        permissions={"external_web.read", "memory.read"},
        required_auth=set(),
        memory_scopes={"global_user", "workspace", "project", "public_profile"},
        tools=["web_search_mcp"],
        cost_per_1k_tokens=0.004,
        p95_latency_ms=4200,
        success_rate=0.91,
        user_success_rate=0.86,
        workspace_success_rate=0.90,
        trust_score=0.88,
        risk_level=0.20,
        verified=True,
        install_count=8200,
        rating=4.7,
    ),
    Agent(
        id="email_agent",
        name="Email Summary Agent",
        description="Reads Gmail and summarizes important unread emails with action items.",
        skills={"gmail_read": 0.96, "writing": 0.78, "memory_retrieval": 0.60},
        permissions={"email.read", "memory.read"},
        required_auth={"google"},
        memory_scopes={"global_user", "workspace", "private_sensitive"},
        tools=["gmail_mcp"],
        cost_per_1k_tokens=0.003,
        p95_latency_ms=3000,
        success_rate=0.93,
        user_success_rate=0.89,
        workspace_success_rate=0.91,
        trust_score=0.90,
        risk_level=0.30,
        verified=True,
        install_count=11000,
        rating=4.8,
    ),
    Agent(
        id="email_send_agent",
        name="Email Send Agent",
        description="Drafts and sends emails with approval, using user preferences and CRM context.",
        skills={"gmail_send": 0.91, "writing": 0.85, "memory_retrieval": 0.65},
        permissions={"email.read", "email.send", "memory.read"},
        required_auth={"google"},
        memory_scopes={"global_user", "workspace", "private_sensitive"},
        tools=["gmail_mcp"],
        cost_per_1k_tokens=0.004,
        p95_latency_ms=3600,
        success_rate=0.89,
        user_success_rate=0.83,
        workspace_success_rate=0.87,
        trust_score=0.84,
        risk_level=0.65,
        verified=True,
        install_count=6200,
        rating=4.5,
    ),
    Agent(
        id="calendar_agent",
        name="Calendar Planning Agent",
        description="Reads calendars, finds availability, and creates events after approval.",
        skills={"calendar_read": 0.94, "calendar_write": 0.90, "memory_retrieval": 0.70},
        permissions={"calendar.read", "calendar.write", "memory.read"},
        required_auth={"google"},
        memory_scopes={"global_user", "workspace", "private_sensitive"},
        tools=["calendar_mcp"],
        cost_per_1k_tokens=0.003,
        p95_latency_ms=2500,
        success_rate=0.95,
        user_success_rate=0.91,
        workspace_success_rate=0.92,
        trust_score=0.91,
        risk_level=0.50,
        verified=True,
        install_count=9900,
        rating=4.9,
    ),
    Agent(
        id="github_agent",
        name="GitHub Triage Agent",
        description="Analyzes GitHub issues and pull requests, labels bugs, and writes triage summaries.",
        skills={"github_triage": 0.96, "writing": 0.70, "memory_retrieval": 0.55},
        permissions={"code.read", "code.write", "memory.read"},
        required_auth={"github"},
        memory_scopes={"workspace", "project", "agent_specific"},
        tools=["github_mcp"],
        cost_per_1k_tokens=0.005,
        p95_latency_ms=5200,
        success_rate=0.88,
        user_success_rate=0.85,
        workspace_success_rate=0.89,
        trust_score=0.86,
        risk_level=0.58,
        verified=True,
        install_count=7600,
        rating=4.6,
    ),
    Agent(
        id="stripe_agent",
        name="Revenue Insights Agent",
        description="Analyzes Stripe revenue, churn, payments, subscriptions, and growth metrics.",
        skills={"stripe_analytics": 0.97, "writing": 0.72, "memory_retrieval": 0.60},
        permissions={"payments.read", "memory.read"},
        required_auth={"stripe"},
        memory_scopes={"workspace", "private_sensitive"},
        tools=["stripe_mcp"],
        cost_per_1k_tokens=0.006,
        p95_latency_ms=4800,
        success_rate=0.90,
        user_success_rate=0.80,
        workspace_success_rate=0.88,
        trust_score=0.87,
        risk_level=0.45,
        verified=True,
        install_count=4800,
        rating=4.5,
    ),
    Agent(
        id="travel_agent",
        name="Travel Search Agent",
        description="Finds flights, hotels, prices, routes, and travel options based on preferences.",
        skills={"travel_search": 0.96, "web_research": 0.70, "memory_retrieval": 0.80},
        permissions={"external_web.read", "memory.read"},
        required_auth=set(),
        memory_scopes={"global_user", "private_sensitive", "project"},
        tools=["travel_search_mcp", "web_search_mcp"],
        cost_per_1k_tokens=0.005,
        p95_latency_ms=6200,
        success_rate=0.86,
        user_success_rate=0.82,
        workspace_success_rate=0.84,
        trust_score=0.78,
        risk_level=0.40,
        verified=True,
        install_count=3400,
        rating=4.4,
    ),
    Agent(
        id="booking_agent",
        name="Travel Booking Agent",
        description="Books flights and hotels after explicit approval and creates booking records.",
        skills={"travel_book": 0.91, "travel_search": 0.80, "memory_retrieval": 0.70},
        permissions={"external_web.read", "payments.write", "memory.read"},
        required_auth={"travel_provider", "payment_method"},
        memory_scopes={"global_user", "private_sensitive"},
        tools=["travel_booking_mcp"],
        cost_per_1k_tokens=0.006,
        p95_latency_ms=7000,
        success_rate=0.81,
        user_success_rate=0.75,
        workspace_success_rate=0.78,
        trust_score=0.76,
        risk_level=0.88,
        verified=False,
        install_count=1200,
        rating=4.0,
    ),
    Agent(
        id="writer_agent",
        name="Writer Agent",
        description="Writes polished reports, summaries, product docs, proposals, and final messages.",
        skills={"writing": 0.97, "memory_retrieval": 0.65, "web_research": 0.50},
        permissions={"memory.read"},
        required_auth=set(),
        memory_scopes={"global_user", "workspace", "project"},
        tools=[],
        cost_per_1k_tokens=0.003,
        p95_latency_ms=2800,
        success_rate=0.92,
        user_success_rate=0.88,
        workspace_success_rate=0.90,
        trust_score=0.89,
        risk_level=0.18,
        verified=True,
        install_count=9100,
        rating=4.8,
    ),
    Agent(
        id="qa_agent",
        name="QA Agent",
        description="Checks answers, validates claims, tests outputs, and finds mistakes before final delivery.",
        skills={"qa_review": 0.96, "writing": 0.50, "web_research": 0.55},
        permissions={"external_web.read", "memory.read"},
        required_auth=set(),
        memory_scopes={"workspace", "project", "public_profile"},
        tools=["web_search_mcp"],
        cost_per_1k_tokens=0.004,
        p95_latency_ms=4500,
        success_rate=0.89,
        user_success_rate=0.86,
        workspace_success_rate=0.88,
        trust_score=0.88,
        risk_level=0.22,
        verified=True,
        install_count=5200,
        rating=4.6,
    ),
]


MEMORIES: List[MemoryItem] = [
    MemoryItem(
        id="m1",
        title="Travel preferences",
        content="User prefers morning flights, aisle seats, direct routes, and hotels near business districts.",
        scope="global_user",
        sensitivity=0.35,
        historical_usefulness=0.90,
        authority=0.90,
    ),
    MemoryItem(
        id="m2",
        title="Calendar work hours",
        content="User prefers meetings between 10 AM and 5 PM India time and avoids Friday evenings.",
        scope="private_sensitive",
        sensitivity=0.70,
        allowed_agents={"calendar_agent", "travel_agent", "booking_agent"},
        historical_usefulness=0.85,
        authority=0.80,
    ),
    MemoryItem(
        id="m3",
        title="GitHub project context",
        content="The SaaS project uses Next.js, Prisma, PostgreSQL, MCP gateway, and agent routing.",
        scope="project",
        sensitivity=0.25,
        allowed_agents={"github_agent", "research_agent", "writer_agent", "qa_agent"},
        historical_usefulness=0.80,
        authority=0.75,
    ),
    MemoryItem(
        id="m4",
        title="Revenue dashboard preference",
        content="Founder wants weekly Stripe revenue summaries with MRR, ARR, churn, refunds, and failed payments.",
        scope="workspace",
        sensitivity=0.65,
        allowed_agents={"stripe_agent", "writer_agent"},
        historical_usefulness=0.75,
        authority=0.80,
    ),
    MemoryItem(
        id="m5",
        title="Writing style",
        content="Prefer clear structured answers with direct recommendations and minimal jargon.",
        scope="public_profile",
        sensitivity=0.10,
        historical_usefulness=0.95,
        authority=0.90,
    ),
]


# ============================================================
# 4. Text vector helpers
# ============================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def cosine_tfidf(query: str, docs: List[str]) -> np.ndarray:
    """Return cosine similarity between query and each document using TF-IDF."""
    corpus = [normalize_text(query)] + [normalize_text(d) for d in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    return cosine_similarity(X[0:1], X[1:]).flatten()


def minmax(values: List[float], default: float = 0.0) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [default for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# ============================================================
# 5. Weighted PageRank / AgentRank
# ============================================================

def weighted_pagerank(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
    priors: Optional[Dict[str, float]] = None,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Weighted PageRank with personalized priors.

    Interpretation:
    - nodes are agents, tools, skills, workflows, publishers, etc.
    - edge src -> dst means src gives authority to dst.
    - priors are teleport probabilities, useful for verified/trusted nodes.
    """
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    if priors is None:
        p = np.ones(n) / n
    else:
        p = np.array([max(float(priors.get(node, 0.0)), 0.0) for node in nodes], dtype=float)
        if p.sum() <= 0:
            p = np.ones(n) / n
        else:
            p = p / p.sum()

    outgoing: Dict[str, List[Tuple[str, float]]] = {node: [] for node in nodes}
    for src, dst, w in edges:
        if src in idx and dst in idx and w > 0:
            outgoing[src].append((dst, float(w)))

    rank = p.copy()

    for iteration in range(max_iter):
        new_rank = (1.0 - damping) * p

        # Distribute authority from each node.
        for src in nodes:
            src_i = idx[src]
            outs = outgoing[src]
            if not outs:
                # Dangling nodes teleport according to prior distribution.
                new_rank += damping * rank[src_i] * p
            else:
                total_w = sum(w for _, w in outs)
                for dst, w in outs:
                    new_rank[idx[dst]] += damping * rank[src_i] * (w / total_w)

        # Numerical safety.
        new_rank = new_rank / new_rank.sum()

        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank

    return {node: float(rank[idx[node]]) for node in nodes}


def build_authority_graph(agents: List[Agent], skills: Dict[str, Skill]) -> Tuple[List[str], List[Tuple[str, str, float]], Dict[str, float]]:
    """Build a heterogeneous graph of agents, skills, tools, and workflows."""
    nodes: Set[str] = set()
    edges: List[Tuple[str, str, float]] = []
    priors: Dict[str, float] = {}

    # Add skill nodes.
    for skill_id, skill in skills.items():
        node = f"skill:{skill_id}"
        nodes.add(node)
        priors[node] = 0.10 + 0.20 * skill.historical_demand

    # Add agent and tool nodes.
    for a in agents:
        agent_node = f"agent:{a.id}"
        nodes.add(agent_node)
        priors[agent_node] = (
            0.20
            + 0.25 * a.trust_score
            + 0.25 * a.success_rate
            + 0.15 * (1.0 if a.verified else 0.0)
            + 0.10 * min(math.log1p(a.install_count) / 10.0, 1.0)
            + 0.05 * (a.rating / 5.0)
        )

        for tool_id in a.tools:
            tool_node = f"tool:{tool_id}"
            nodes.add(tool_node)
            priors.setdefault(tool_node, 0.15)

            # A reliable tool endorses agents that use it successfully.
            edges.append((tool_node, agent_node, 0.30 * a.success_rate + 0.20 * a.trust_score))
            # Agent also endorses the tools it uses.
            edges.append((agent_node, tool_node, 0.25 * a.success_rate))

        for skill_id, prof in a.skills.items():
            skill_node = f"skill:{skill_id}"
            # Skill endorses agents that are good at it.
            edges.append((skill_node, agent_node, 0.60 * prof + 0.20 * a.success_rate + 0.20 * a.trust_score))
            # Agent contributes evidence back to skill category.
            edges.append((agent_node, skill_node, 0.40 * prof))

    # Add workflow-like edges representing successful agent-to-agent handoffs.
    handoffs = [
        ("research_agent", "writer_agent", 0.90),
        ("writer_agent", "qa_agent", 0.85),
        ("travel_agent", "booking_agent", 0.70),
        ("booking_agent", "calendar_agent", 0.80),
        ("email_agent", "calendar_agent", 0.55),
        ("github_agent", "writer_agent", 0.65),
        ("stripe_agent", "writer_agent", 0.60),
    ]
    for src, dst, w in handoffs:
        edges.append((f"agent:{src}", f"agent:{dst}", w))

    return sorted(nodes), edges, priors


def compute_agent_rank(agents: List[Agent], skills: Dict[str, Skill]) -> Dict[str, float]:
    nodes, edges, priors = build_authority_graph(agents, skills)
    scores = weighted_pagerank(nodes, edges, priors=priors, damping=0.85)
    agent_scores = {a.id: scores.get(f"agent:{a.id}", 0.0) for a in agents}
    total = sum(agent_scores.values()) or 1.0
    return {k: v / total for k, v in agent_scores.items()}


# ============================================================
# 6. SkillRank: rank skills for a task
# ============================================================

def keyword_overlap_score(task: str, keywords: List[str]) -> float:
    t = normalize_text(task)
    hits = 0
    for kw in keywords:
        if normalize_text(kw) in t:
            hits += 1
    return min(hits / max(len(keywords), 1), 1.0)


def rank_skills_for_task(task: str, agents: List[Agent], skills: Dict[str, Skill], agent_rank: Dict[str, float]) -> pd.DataFrame:
    skill_ids = list(skills.keys())
    skill_docs = [skills[s].description + " " + " ".join(skills[s].keywords) for s in skill_ids]
    sem = cosine_tfidf(task, skill_docs)

    rows = []
    for skill_id, semantic_score in zip(skill_ids, sem):
        skill = skills[skill_id]
        keyword_score = keyword_overlap_score(task, skill.keywords)

        # Supply quality: how many highly ranked agents can perform this skill?
        supply_quality = 0.0
        for a in agents:
            supply_quality += agent_rank.get(a.id, 0.0) * a.skills.get(skill_id, 0.0)
        supply_quality = min(supply_quality * 5.0, 1.0)  # scale for visibility

        score = (
            0.45 * float(semantic_score)
            + 0.25 * keyword_score
            + 0.15 * skill.historical_demand
            + 0.15 * supply_quality
        )

        rows.append({
            "skill_id": skill_id,
            "skill": skill.name,
            "score": score,
            "semantic": float(semantic_score),
            "keyword": keyword_score,
            "demand": skill.historical_demand,
            "supply_quality": supply_quality,
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def infer_required_permissions(top_skill_ids: List[str]) -> Set[str]:
    mapping = {
        "gmail_read": {"email.read"},
        "gmail_send": {"email.read", "email.send"},
        "calendar_read": {"calendar.read"},
        "calendar_write": {"calendar.read", "calendar.write"},
        "github_triage": {"code.read"},
        "stripe_analytics": {"payments.read"},
        "travel_search": {"external_web.read"},
        "travel_book": {"external_web.read", "payments.write"},
        "web_research": {"external_web.read"},
        "writing": set(),
        "qa_review": {"external_web.read"},
        "memory_retrieval": {"memory.read"},
    }
    perms: Set[str] = set()
    for s in top_skill_ids:
        perms |= mapping.get(s, set())
    return perms


def infer_required_auth(top_skill_ids: List[str]) -> Set[str]:
    mapping = {
        "gmail_read": {"google"},
        "gmail_send": {"google"},
        "calendar_read": {"google"},
        "calendar_write": {"google"},
        "github_triage": {"github"},
        "stripe_analytics": {"stripe"},
        "travel_book": {"travel_provider", "payment_method"},
    }
    auth: Set[str] = set()
    for s in top_skill_ids:
        auth |= mapping.get(s, set())
    return auth


# ============================================================
# 7. Memory scoring
# ============================================================

def memory_policy_allows(m: MemoryItem, agent: Agent, context: RankingContext) -> bool:
    if m.scope not in context.allowed_memory_scopes:
        return False
    if agent.id in m.blocked_agents:
        return False
    if m.allowed_agents and agent.id not in m.allowed_agents:
        return False
    if context.client_name in m.blocked_clients:
        return False
    if m.allowed_clients and context.client_name not in m.allowed_clients:
        return False
    # Agent must be allowed to read this memory scope.
    return m.scope in agent.memory_scopes


def memory_fit_score(task: str, agent: Agent, context: RankingContext, memories: List[MemoryItem]) -> float:
    allowed = [m for m in memories if memory_policy_allows(m, agent, context)]
    if not allowed:
        return 0.0

    docs = [m.title + " " + m.content for m in allowed]
    sims = cosine_tfidf(task, docs)
    scores = []
    for m, sim in zip(allowed, sims):
        # New memory scoring formula.
        score = (
            0.35 * float(sim)
            + 0.20 * (1.0 if m.scope in agent.memory_scopes else 0.0)
            + 0.15 * m.historical_usefulness
            + 0.10 * 0.75  # recency placeholder
            + 0.10 * m.authority
            + 0.10 * keyword_overlap_score(task, [m.title])
            - 0.20 * max(m.sensitivity - (1.0 - agent.risk_level), 0.0)
        )
        scores.append(max(score, 0.0))
    return min(max(scores), 1.0)


# ============================================================
# 8. AURA-Rank: task-specific agent ranking
# ============================================================

def capability_match(agent: Agent, skill_rank_df: pd.DataFrame, top_k: int = 4) -> float:
    required = skill_rank_df.head(top_k)[["skill_id", "score"]].values.tolist()
    total_weight = sum(float(w) for _, w in required) or 1.0
    matched = 0.0
    for skill_id, w in required:
        prof = agent.skills.get(skill_id, 0.0)
        matched += float(w) * prof
    return min(matched / total_weight, 1.0)


def permission_fit(agent: Agent, context: RankingContext, required_permissions: Set[str]) -> float:
    if not required_permissions:
        return 1.0
    agent_ok = len(required_permissions & agent.permissions) / len(required_permissions)
    client_ok = len(required_permissions & context.client_permissions) / len(required_permissions)
    return min(agent_ok, client_ok)


def auth_readiness(agent: Agent, context: RankingContext, required_auth: Set[str]) -> float:
    needed = agent.required_auth | required_auth
    if not needed:
        return 1.0
    return len(needed & context.connected_auth) / len(needed)


def latency_score(agent: Agent, context: RankingContext) -> float:
    return float(math.exp(-agent.p95_latency_ms / max(context.target_latency_ms, 1)))


def cost_score(agent: Agent, context: RankingContext, estimated_tokens: int = 3000) -> float:
    estimated_cost = (estimated_tokens / 1000.0) * agent.cost_per_1k_tokens
    return 1.0 - min(estimated_cost / max(context.budget_per_run_usd, 1e-9), 1.0)


def risk_penalty(agent: Agent, required_permissions: Set[str]) -> float:
    high_risk_perms = {"payments.write", "email.send", "calendar.write", "code.write", "files.write"}
    action_risk = 1.0 if required_permissions & high_risk_perms else 0.0
    return min(0.70 * agent.risk_level + 0.30 * action_risk, 1.0)


def semantic_agent_match(task: str, agent: Agent) -> float:
    skill_text = " ".join([SKILLS[s].name + " " + SKILLS[s].description for s in agent.skills if s in SKILLS])
    doc = f"{agent.name} {agent.description} {skill_text}"
    return float(cosine_tfidf(task, [doc])[0])


def aura_rank_agents(
    task: str,
    agents: List[Agent],
    skills: Dict[str, Skill],
    memories: List[MemoryItem],
    context: RankingContext,
    agent_rank: Optional[Dict[str, float]] = None,
    top_skill_k: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    AURA-Rank: Agent Utility and Reputation-aware Adaptive Rank.

    Score(agent | task, user, workspace, client) =
        graph authority + skill fit + operational readiness - risk.
    """
    if agent_rank is None:
        agent_rank = compute_agent_rank(agents, skills)

    skill_df = rank_skills_for_task(task, agents, skills, agent_rank)
    top_skills = skill_df.head(top_skill_k)["skill_id"].tolist()
    required_permissions = infer_required_permissions(top_skills)
    required_auth = infer_required_auth(top_skills)

    rows = []
    for a in agents:
        features = {
            "semantic_task_match": semantic_agent_match(task, a),
            "capability_match": capability_match(a, skill_df, top_k=top_skill_k),
            "agent_rank": agent_rank.get(a.id, 0.0),
            "user_history_success": a.user_success_rate,
            "workspace_history_success": a.workspace_success_rate,
            "auth_readiness": auth_readiness(a, context, required_auth),
            "permission_fit": permission_fit(a, context, required_permissions),
            "latency_score": latency_score(a, context),
            "cost_score": cost_score(a, context),
            "memory_fit": memory_fit_score(task, a, context, memories),
            "trust_score": a.trust_score,
            "risk_penalty": risk_penalty(a, required_permissions),
            "missing_scope_penalty": 1.0 - permission_fit(a, context, required_permissions),
        }

        score = (
            0.25 * features["semantic_task_match"]
            + 0.18 * features["capability_match"]
            + 0.12 * features["agent_rank"]
            + 0.10 * features["user_history_success"]
            + 0.08 * features["workspace_history_success"]
            + 0.08 * features["auth_readiness"]
            + 0.07 * features["permission_fit"]
            + 0.06 * features["latency_score"]
            + 0.06 * features["cost_score"]
            + 0.05 * features["memory_fit"]
            + 0.05 * features["trust_score"]
            - 0.15 * features["risk_penalty"]
            - 0.10 * features["missing_scope_penalty"]
        )

        row = {
            "agent_id": a.id,
            "agent": a.name,
            "score": float(score),
            "top_skills": ", ".join(top_skills),
            "required_permissions": ", ".join(sorted(required_permissions)),
            "required_auth": ", ".join(sorted(required_auth)),
        }
        row.update(features)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df, skill_df


# ============================================================
# 9. Evaluation metrics
# ============================================================

def hit_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    return 1.0 if any(a in relevant_ids for a in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, agent_id in enumerate(ranked_ids, start=1):
        if agent_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for i, agent_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if agent_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_rels = [1.0] * min(len(relevant_ids), k)
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranker(test_cases: List[Dict[str, Any]], context: RankingContext) -> pd.DataFrame:
    agent_rank = compute_agent_rank(AGENTS, SKILLS)
    rows = []
    for tc in test_cases:
        df, skill_df = aura_rank_agents(tc["task"], AGENTS, SKILLS, MEMORIES, context, agent_rank)
        ranked_ids = df["agent_id"].tolist()
        relevant = set(tc["expected_agents"])
        rows.append({
            "task": tc["task"],
            "top_1": ranked_ids[0],
            "expected": ", ".join(tc["expected_agents"]),
            "hit@1": hit_at_k(ranked_ids, relevant, 1),
            "hit@3": hit_at_k(ranked_ids, relevant, 3),
            "mrr": reciprocal_rank(ranked_ids, relevant),
            "ndcg@3": ndcg_at_k(ranked_ids, relevant, 3),
            "top_skills": ", ".join(skill_df.head(3)["skill_id"].tolist()),
        })
    return pd.DataFrame(rows)


# ============================================================
# 10. A2A ChainRank
# ============================================================

def decompose_task_to_subtasks(task: str) -> List[str]:
    """Simple heuristic task decomposition for prototype."""
    t = normalize_text(task)
    subtasks = []
    if any(x in t for x in ["research", "find", "compare", "search"]):
        subtasks.append("Research and gather options for: " + task)
    if any(x in t for x in ["book", "ticket", "flight", "hotel", "reservation"]):
        subtasks.append("Book or prepare booking after approval for: " + task)
    if any(x in t for x in ["calendar", "schedule", "meeting", "event"]):
        subtasks.append("Update calendar or schedule event for: " + task)
    if any(x in t for x in ["write", "summary", "report", "draft"]):
        subtasks.append("Write final summary or report for: " + task)
    if not subtasks:
        subtasks = [task]
    # Always add QA for complex multi-step tasks.
    if len(subtasks) >= 2:
        subtasks.append("Review and validate the final result for: " + task)
    return subtasks


def handoff_compatibility(agent_a: str, agent_b: str) -> float:
    known = {
        ("research_agent", "writer_agent"): 0.90,
        ("writer_agent", "qa_agent"): 0.88,
        ("travel_agent", "booking_agent"): 0.78,
        ("booking_agent", "calendar_agent"): 0.82,
        ("calendar_agent", "email_agent"): 0.62,
        ("github_agent", "writer_agent"): 0.70,
        ("stripe_agent", "writer_agent"): 0.65,
    }
    return known.get((agent_a, agent_b), 0.50)


def plan_a2a_chain(task: str, context: RankingContext, beam_width: int = 3) -> pd.DataFrame:
    agent_rank = compute_agent_rank(AGENTS, SKILLS)
    subtasks = decompose_task_to_subtasks(task)

    # Top candidates per subtask.
    candidates_by_step = []
    for subtask in subtasks:
        df, _ = aura_rank_agents(subtask, AGENTS, SKILLS, MEMORIES, context, agent_rank)
        candidates_by_step.append(df.head(beam_width)[["agent_id", "agent", "score", "risk_penalty", "latency_score", "cost_score"]])

    # Beam search.
    beams = [([], 0.0, [])]  # path, score, details
    for step_i, cand_df in enumerate(candidates_by_step):
        new_beams = []
        for path, current_score, details in beams:
            for _, row in cand_df.iterrows():
                agent_id = row["agent_id"]
                handoff = 1.0 if not path else handoff_compatibility(path[-1], agent_id)
                incremental = float(row["score"]) + 0.10 * handoff
                new_path = path + [agent_id]
                new_score = current_score + incremental
                new_details = details + [{
                    "step": step_i + 1,
                    "subtask": subtasks[step_i],
                    "agent_id": agent_id,
                    "agent": row["agent"],
                    "agent_score": float(row["score"]),
                    "handoff": handoff,
                    "step_score": incremental,
                }]
                new_beams.append((new_path, new_score, new_details))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    best_path, best_score, best_details = beams[0]
    result = pd.DataFrame(best_details)
    result["chain_score"] = best_score
    return result


# ============================================================
# 11. Demonstration
# ============================================================

def print_ranking(task: str, context: RankingContext, top_n: int = 5) -> None:
    agent_rank = compute_agent_rank(AGENTS, SKILLS)
    df, skill_df = aura_rank_agents(task, AGENTS, SKILLS, MEMORIES, context, agent_rank)

    print("\n" + "=" * 90)
    print("TASK:", task)
    print("=" * 90)
    print("\nTop skills inferred:")
    print(skill_df.head(6).to_string(index=False))

    cols = [
        "agent", "score", "semantic_task_match", "capability_match", "agent_rank",
        "auth_readiness", "permission_fit", "memory_fit", "cost_score",
        "latency_score", "risk_penalty"
    ]
    print("\nTop agents:")
    print(df[cols].head(top_n).to_string(index=False))

    best = df.iloc[0]
    print("\nSelected agent:", best["agent"])
    print("Reason: high combination of task match, capability, permission/auth readiness, memory fit, and low risk.")


def plot_ranking(df: pd.DataFrame, title: str = "Agent ranking") -> None:
    if plt is None:
        print("matplotlib not available")
        return
    top = df.head(8).copy()
    plt.figure(figsize=(10, 5))
    plt.barh(top["agent"], top["score"])
    plt.gca().invert_yaxis()
    plt.xlabel("AURA-Rank score")
    plt.title(title)
    plt.show()


def main_demo():
    context = RankingContext(
        user_id="u_demo",
        workspace_id="w_demo",
        client_name="claude",
        client_permissions={
            "external_web.read", "memory.read", "email.read", "calendar.read", "calendar.write",
            "payments.read", "code.read", "code.write", "payments.write"
        },
        connected_auth={"google", "github", "stripe", "travel_provider", "payment_method"},
        allowed_memory_scopes={"global_user", "workspace", "project", "private_sensitive", "public_profile"},
        budget_per_run_usd=0.20,
        target_latency_ms=5000,
    )

    tasks = [
        "Book a flight from Hyderabad to Mumbai next Friday morning and add it to my calendar.",
        "Summarize my unread Gmail messages and extract action items.",
        "Analyze open GitHub issues and create a triage summary.",
        "Give me a Stripe revenue report with MRR, ARR, churn and refunds.",
        "Research competitors and write a short market brief with citations.",
    ]

    for task in tasks:
        print_ranking(task, context, top_n=5)

    # Evaluation.
    test_cases = [
        {"task": "Summarize unread Gmail messages", "expected_agents": ["email_agent"]},
        {"task": "Create a calendar event for my meeting tomorrow", "expected_agents": ["calendar_agent"]},
        {"task": "Analyze GitHub bugs and label issues", "expected_agents": ["github_agent"]},
        {"task": "Show Stripe MRR and revenue summary", "expected_agents": ["stripe_agent"]},
        {"task": "Find flights and hotels for Mumbai trip", "expected_agents": ["travel_agent", "booking_agent"]},
        {"task": "Write a polished report from this research", "expected_agents": ["writer_agent"]},
        {"task": "Review this answer and check if it is correct", "expected_agents": ["qa_agent"]},
    ]
    eval_df = evaluate_ranker(test_cases, context)
    print("\n" + "=" * 90)
    print("EVALUATION")
    print("=" * 90)
    print(eval_df.to_string(index=False))
    print("\nMean metrics:")
    print(eval_df[["hit@1", "hit@3", "mrr", "ndcg@3"]].mean().to_string())

    # A2A chain plan.
    chain_task = "Research travel options, book the best flight to Mumbai, add it to calendar, and write a summary."
    chain_df = plan_a2a_chain(chain_task, context)
    print("\n" + "=" * 90)
    print("A2A CHAIN PLAN")
    print("=" * 90)
    print(chain_df.to_string(index=False))

    # Plot one ranking.
    df, _ = aura_rank_agents(tasks[0], AGENTS, SKILLS, MEMORIES, context, compute_agent_rank(AGENTS, SKILLS))
    plot_ranking(df, title="AURA-Rank for travel + calendar task")


if __name__ == "__main__":
    main_demo()
