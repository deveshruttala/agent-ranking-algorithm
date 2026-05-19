
"""
AURA-Rank Real Usecase Codebase for Universal MCP Gateway
=========================================================

Goal:
Provide production-style Python code that simulates the routing engine inside
your SaaS MCP Gateway.

What this code demonstrates:
- One MCP gateway receives a task from any LLM client.
- The router searches scoped memory.
- The router ranks skills and agents.
- The router checks auth, permissions, risk, and budget.
- The router chooses an agent or A2A chain.
- The engine writes logs, tool calls, token estimates, cost, and approval state.

This is still a mock runtime, but the service boundaries are close to what you
would implement in your real SaaS backend.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    raise RuntimeError("Install sklearn in Colab: !pip -q install scikit-learn") from e


# ============================================================
# 1. Domain models
# ============================================================

@dataclass
class ClientToken:
    token_id: str
    workspace_id: str
    user_id: str
    client_name: str
    allowed_permissions: Set[str]
    allowed_memory_scopes: Set[str]
    rate_limit_per_minute: int = 60
    revoked: bool = False


@dataclass
class Connection:
    provider: str
    status: str  # connected, expired, missing
    scopes: Set[str]


@dataclass
class MemoryItem:
    id: str
    workspace_id: str
    title: str
    content: str
    scope: str
    sensitivity: float
    allowed_agents: Set[str] = field(default_factory=set)
    blocked_agents: Set[str] = field(default_factory=set)
    historical_usefulness: float = 0.7
    authority: float = 0.7


@dataclass
class ToolSpec:
    id: str
    name: str
    provider: str
    required_permissions: Set[str]
    required_auth: Set[str]
    risk_level: float
    description: str


@dataclass
class AgentSpec:
    id: str
    name: str
    description: str
    system_instructions: str
    skill_proficiency: Dict[str, float]
    allowed_tools: Set[str]
    permissions: Set[str]
    required_auth: Set[str]
    memory_scopes: Set[str]
    success_rate: float
    trust_score: float
    risk_level: float
    avg_cost_usd: float
    p95_latency_ms: int
    install_count: int
    rating: float
    verified: bool


@dataclass
class ExecutionStep:
    step_id: str
    agent_id: str
    agent_name: str
    input: str
    output: str
    tools_used: List[str]
    memory_used: List[str]
    tokens_estimated: int
    cost_estimated: float
    latency_ms: int
    status: str
    approval_required: bool = False


@dataclass
class ExecutionResult:
    execution_id: str
    workspace_id: str
    user_id: str
    client_name: str
    task: str
    selected_agent_ids: List[str]
    status: str
    approval_required: bool
    approval_reason: Optional[str]
    total_tokens_estimated: int
    total_cost_estimated: float
    total_latency_ms: int
    steps: List[ExecutionStep]
    ranking_table: List[Dict[str, Any]]


# ============================================================
# 2. Seed data
# ============================================================

SKILL_DEFINITIONS: Dict[str, str] = {
    "web_research": "Search web, gather facts, compare sources, cite findings.",
    "gmail_read": "Read and summarize Gmail messages.",
    "gmail_send": "Draft and send Gmail messages after approval.",
    "calendar_read": "Read events and availability from calendar.",
    "calendar_write": "Create or update calendar events.",
    "github_triage": "Analyze GitHub issues, pull requests, code and labels.",
    "stripe_analytics": "Analyze revenue, MRR, ARR, refunds and Stripe payments.",
    "travel_search": "Search flights, hotels, routes and travel prices.",
    "travel_book": "Book flights, hotels and reservations after approval.",
    "writing": "Write reports, summaries, proposals and messages.",
    "qa_review": "Review outputs, validate claims, check quality and errors.",
    "memory_retrieval": "Retrieve scoped user, project, workspace or agent memory.",
}

SKILL_KEYWORDS: Dict[str, List[str]] = {
    "web_research": ["research", "search", "find", "competitor", "market", "source"],
    "gmail_read": ["email", "gmail", "inbox", "unread", "summarize email"],
    "gmail_send": ["send email", "reply", "draft email", "outreach"],
    "calendar_read": ["calendar", "availability", "schedule", "meeting"],
    "calendar_write": ["add to calendar", "create event", "schedule meeting", "update calendar"],
    "github_triage": ["github", "issue", "pull request", "repo", "bug", "triage"],
    "stripe_analytics": ["stripe", "revenue", "mrr", "arr", "refund", "payment", "churn"],
    "travel_search": ["travel", "flight", "hotel", "trip", "itinerary", "route"],
    "travel_book": ["book", "ticket", "reservation", "purchase flight", "book hotel"],
    "writing": ["write", "draft", "report", "summary", "brief"],
    "qa_review": ["review", "qa", "validate", "test", "check"],
    "memory_retrieval": ["remember", "preference", "memory", "context"],
}

PERMISSION_BY_SKILL: Dict[str, Set[str]] = {
    "web_research": {"external_web.read"},
    "gmail_read": {"email.read"},
    "gmail_send": {"email.read", "email.send"},
    "calendar_read": {"calendar.read"},
    "calendar_write": {"calendar.read", "calendar.write"},
    "github_triage": {"code.read"},
    "stripe_analytics": {"payments.read"},
    "travel_search": {"external_web.read"},
    "travel_book": {"external_web.read", "payments.write"},
    "writing": set(),
    "qa_review": {"external_web.read"},
    "memory_retrieval": {"memory.read"},
}

AUTH_BY_SKILL: Dict[str, Set[str]] = {
    "gmail_read": {"google"},
    "gmail_send": {"google"},
    "calendar_read": {"google"},
    "calendar_write": {"google"},
    "github_triage": {"github"},
    "stripe_analytics": {"stripe"},
    "travel_book": {"travel_provider", "payment_method"},
}

TOOLS: Dict[str, ToolSpec] = {
    "web_search_mcp": ToolSpec("web_search_mcp", "Web Search MCP", "web", {"external_web.read"}, set(), 0.20, "Search the web."),
    "gmail_mcp": ToolSpec("gmail_mcp", "Gmail MCP", "google", {"email.read", "email.send"}, {"google"}, 0.55, "Read and send Gmail."),
    "calendar_mcp": ToolSpec("calendar_mcp", "Calendar MCP", "google", {"calendar.read", "calendar.write"}, {"google"}, 0.50, "Manage calendar events."),
    "github_mcp": ToolSpec("github_mcp", "GitHub MCP", "github", {"code.read", "code.write"}, {"github"}, 0.60, "Analyze and update GitHub."),
    "stripe_mcp": ToolSpec("stripe_mcp", "Stripe MCP", "stripe", {"payments.read"}, {"stripe"}, 0.45, "Analyze Stripe revenue."),
    "travel_search_mcp": ToolSpec("travel_search_mcp", "Travel Search MCP", "travel", {"external_web.read"}, set(), 0.35, "Search travel options."),
    "travel_booking_mcp": ToolSpec("travel_booking_mcp", "Travel Booking MCP", "travel", {"external_web.read", "payments.write"}, {"travel_provider", "payment_method"}, 0.90, "Book travel after approval."),
}

AGENTS: Dict[str, AgentSpec] = {
    "research_agent": AgentSpec(
        "research_agent", "Personal Research Agent", "Finds reliable info and summarizes it.",
        "Be accurate, compare sources, and produce concise findings.",
        {"web_research": 0.96, "writing": 0.72, "memory_retrieval": 0.75},
        {"web_search_mcp"}, {"external_web.read", "memory.read"}, set(),
        {"global_user", "workspace", "project", "public_profile"},
        0.91, 0.88, 0.20, 0.012, 4200, 8200, 4.7, True,
    ),
    "email_agent": AgentSpec(
        "email_agent", "Email Summary Agent", "Summarizes unread Gmail and action items.",
        "Read email only. Never send unless explicit permission is present.",
        {"gmail_read": 0.96, "writing": 0.76, "memory_retrieval": 0.65},
        {"gmail_mcp"}, {"email.read", "memory.read"}, {"google"},
        {"global_user", "workspace", "private_sensitive"},
        0.93, 0.90, 0.30, 0.010, 3000, 11000, 4.8, True,
    ),
    "calendar_agent": AgentSpec(
        "calendar_agent", "Calendar Planning Agent", "Reads and creates calendar events.",
        "Check preferences before scheduling. Ask before writing.",
        {"calendar_read": 0.94, "calendar_write": 0.90, "memory_retrieval": 0.72},
        {"calendar_mcp"}, {"calendar.read", "calendar.write", "memory.read"}, {"google"},
        {"global_user", "workspace", "private_sensitive"},
        0.95, 0.91, 0.50, 0.009, 2500, 9900, 4.9, True,
    ),
    "github_agent": AgentSpec(
        "github_agent", "GitHub Triage Agent", "Analyzes GitHub issues and PRs.",
        "Inspect issues and PRs. Do not write code without approval.",
        {"github_triage": 0.96, "writing": 0.70, "memory_retrieval": 0.55},
        {"github_mcp"}, {"code.read", "code.write", "memory.read"}, {"github"},
        {"workspace", "project", "agent_specific"},
        0.88, 0.86, 0.58, 0.015, 5200, 7600, 4.6, True,
    ),
    "stripe_agent": AgentSpec(
        "stripe_agent", "Revenue Insights Agent", "Reports Stripe revenue and churn.",
        "Read-only revenue analysis. Never modify payments.",
        {"stripe_analytics": 0.97, "writing": 0.72, "memory_retrieval": 0.60},
        {"stripe_mcp"}, {"payments.read", "memory.read"}, {"stripe"},
        {"workspace", "private_sensitive"},
        0.90, 0.87, 0.45, 0.018, 4800, 4800, 4.5, True,
    ),
    "travel_agent": AgentSpec(
        "travel_agent", "Travel Search Agent", "Finds travel options based on preferences.",
        "Find options. Do not purchase. Use user travel preferences.",
        {"travel_search": 0.96, "web_research": 0.70, "memory_retrieval": 0.82},
        {"travel_search_mcp", "web_search_mcp"}, {"external_web.read", "memory.read"}, set(),
        {"global_user", "private_sensitive", "project"},
        0.86, 0.78, 0.40, 0.014, 6200, 3400, 4.4, True,
    ),
    "booking_agent": AgentSpec(
        "booking_agent", "Travel Booking Agent", "Books flights/hotels after approval.",
        "Prepare booking and require explicit approval before purchase.",
        {"travel_book": 0.91, "travel_search": 0.80, "memory_retrieval": 0.70},
        {"travel_booking_mcp"}, {"external_web.read", "payments.write", "memory.read"}, {"travel_provider", "payment_method"},
        {"global_user", "private_sensitive"},
        0.81, 0.76, 0.88, 0.020, 7000, 1200, 4.0, False,
    ),
    "writer_agent": AgentSpec(
        "writer_agent", "Writer Agent", "Writes polished reports and summaries.",
        "Produce clear structured output in the user's preferred style.",
        {"writing": 0.97, "memory_retrieval": 0.65, "web_research": 0.50},
        set(), {"memory.read"}, set(),
        {"global_user", "workspace", "project", "public_profile"},
        0.92, 0.89, 0.18, 0.008, 2800, 9100, 4.8, True,
    ),
    "qa_agent": AgentSpec(
        "qa_agent", "QA Agent", "Validates outputs and catches errors.",
        "Check correctness, risk, factuality, and missing details.",
        {"qa_review": 0.96, "web_research": 0.55, "writing": 0.50},
        {"web_search_mcp"}, {"external_web.read", "memory.read"}, set(),
        {"workspace", "project", "public_profile"},
        0.89, 0.88, 0.22, 0.011, 4500, 5200, 4.6, True,
    ),
}

MEMORY: List[MemoryItem] = [
    MemoryItem("mem_travel", "w1", "Travel preferences", "User prefers morning flights, aisle seats, direct routes, and hotels near business districts.", "global_user", 0.35),
    MemoryItem("mem_calendar", "w1", "Calendar preference", "Avoid Friday evening meetings. Prefer meetings between 10 AM and 5 PM India time.", "private_sensitive", 0.70, allowed_agents={"calendar_agent", "travel_agent", "booking_agent"}),
    MemoryItem("mem_stack", "w1", "MCP SaaS project context", "Project uses Next.js, Prisma, PostgreSQL, MCP gateway, shared memory, and AURA agent ranking.", "project", 0.25, allowed_agents={"github_agent", "research_agent", "writer_agent", "qa_agent"}),
    MemoryItem("mem_revenue", "w1", "Revenue report preference", "Weekly revenue report should include MRR, ARR, churn, refunds, and failed payments.", "workspace", 0.60, allowed_agents={"stripe_agent", "writer_agent"}),
    MemoryItem("mem_style", "w1", "Writing style", "Prefer clear structured answers with direct recommendations and minimal jargon.", "public_profile", 0.10),
]

CONNECTIONS: Dict[str, Connection] = {
    "google": Connection("google", "connected", {"email.read", "calendar.read", "calendar.write"}),
    "github": Connection("github", "connected", {"code.read", "code.write"}),
    "stripe": Connection("stripe", "connected", {"payments.read"}),
    "travel_provider": Connection("travel_provider", "connected", {"travel.search", "travel.book"}),
    "payment_method": Connection("payment_method", "connected", {"payments.write"}),
}

CLIENT_TOKEN = ClientToken(
    token_id="mcp_tok_demo",
    workspace_id="w1",
    user_id="u1",
    client_name="cursor",
    allowed_permissions={
        "external_web.read", "memory.read", "email.read", "calendar.read", "calendar.write",
        "code.read", "code.write", "payments.read", "payments.write"
    },
    allowed_memory_scopes={"global_user", "workspace", "project", "private_sensitive", "public_profile"},
)


# ============================================================
# 3. Utility functions
# ============================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tfidf_similarity(query: str, docs: List[str]) -> np.ndarray:
    corpus = [normalize_text(query)] + [normalize_text(d) for d in docs]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(corpus)
    return cosine_similarity(X[0:1], X[1:]).flatten()


def contains_keyword(task: str, keywords: List[str]) -> float:
    t = normalize_text(task)
    hits = sum(1 for kw in keywords if normalize_text(kw) in t)
    return min(hits / max(len(keywords), 1), 1.0)


def weighted_pagerank(nodes: List[str], edges: List[Tuple[str, str, float]], priors: Dict[str, float], damping: float = 0.85) -> Dict[str, float]:
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    p = np.array([max(priors.get(node, 0.0), 0.0) for node in nodes], dtype=float)
    p = p / p.sum() if p.sum() else np.ones(n) / n
    rank = p.copy()
    outgoing: Dict[str, List[Tuple[str, float]]] = {node: [] for node in nodes}
    for src, dst, w in edges:
        if src in idx and dst in idx and w > 0:
            outgoing[src].append((dst, w))
    for _ in range(100):
        new = (1.0 - damping) * p
        for src in nodes:
            outs = outgoing[src]
            if not outs:
                new += damping * rank[idx[src]] * p
            else:
                total = sum(w for _, w in outs)
                for dst, w in outs:
                    new[idx[dst]] += damping * rank[idx[src]] * (w / total)
        new = new / new.sum()
        if np.abs(new - rank).sum() < 1e-10:
            rank = new
            break
        rank = new
    return {node: float(rank[idx[node]]) for node in nodes}


def build_agent_rank() -> Dict[str, float]:
    nodes: Set[str] = set()
    edges: List[Tuple[str, str, float]] = []
    priors: Dict[str, float] = {}
    for skill_id in SKILL_DEFINITIONS:
        node = f"skill:{skill_id}"
        nodes.add(node)
        priors[node] = 0.20
    for tool_id, tool in TOOLS.items():
        node = f"tool:{tool_id}"
        nodes.add(node)
        priors[node] = 0.15
    for a in AGENTS.values():
        node = f"agent:{a.id}"
        nodes.add(node)
        priors[node] = 0.20 + 0.25 * a.trust_score + 0.25 * a.success_rate + 0.15 * (1 if a.verified else 0) + 0.10 * min(math.log1p(a.install_count) / 10.0, 1.0)
        for skill_id, prof in a.skill_proficiency.items():
            edges.append((f"skill:{skill_id}", node, 0.6 * prof + 0.2 * a.success_rate + 0.2 * a.trust_score))
            edges.append((node, f"skill:{skill_id}", 0.4 * prof))
        for tool_id in a.allowed_tools:
            edges.append((f"tool:{tool_id}", node, 0.4 * a.success_rate + 0.2 * a.trust_score))
            edges.append((node, f"tool:{tool_id}", 0.3 * a.success_rate))
    handoffs = [
        ("research_agent", "writer_agent", 0.90),
        ("writer_agent", "qa_agent", 0.88),
        ("travel_agent", "booking_agent", 0.78),
        ("booking_agent", "calendar_agent", 0.82),
        ("github_agent", "writer_agent", 0.70),
        ("stripe_agent", "writer_agent", 0.65),
    ]
    for src, dst, w in handoffs:
        edges.append((f"agent:{src}", f"agent:{dst}", w))
    nodes_list = sorted(nodes)
    scores = weighted_pagerank(nodes_list, edges, priors)
    agent_scores = {a_id: scores.get(f"agent:{a_id}", 0.0) for a_id in AGENTS}
    total = sum(agent_scores.values()) or 1.0
    return {k: v / total for k, v in agent_scores.items()}


# ============================================================
# 4. Services
# ============================================================

class PermissionEngine:
    HIGH_RISK_PERMISSIONS = {"payments.write", "email.send", "calendar.write", "code.write", "files.write"}

    @staticmethod
    def required_permissions(skills: List[str]) -> Set[str]:
        perms: Set[str] = set()
        for skill in skills:
            perms |= PERMISSION_BY_SKILL.get(skill, set())
        return perms

    @staticmethod
    def required_auth(skills: List[str]) -> Set[str]:
        auth: Set[str] = set()
        for skill in skills:
            auth |= AUTH_BY_SKILL.get(skill, set())
        return auth

    @staticmethod
    def permission_fit(agent: AgentSpec, token: ClientToken, required_permissions: Set[str]) -> float:
        if not required_permissions:
            return 1.0
        agent_fit = len(required_permissions & agent.permissions) / len(required_permissions)
        client_fit = len(required_permissions & token.allowed_permissions) / len(required_permissions)
        return min(agent_fit, client_fit)

    @staticmethod
    def approval_required(agent: AgentSpec, required_permissions: Set[str]) -> Tuple[bool, Optional[str]]:
        risky = required_permissions & PermissionEngine.HIGH_RISK_PERMISSIONS
        if risky or agent.risk_level >= 0.75:
            return True, f"High-risk action requires approval. Permissions: {sorted(risky)}; agent risk={agent.risk_level:.2f}"
        return False, None


class MemoryStore:
    def __init__(self, memory_items: List[MemoryItem]):
        self.memory_items = memory_items

    def _allowed(self, m: MemoryItem, agent: AgentSpec, token: ClientToken) -> bool:
        if m.workspace_id != token.workspace_id:
            return False
        if m.scope not in token.allowed_memory_scopes:
            return False
        if m.scope not in agent.memory_scopes:
            return False
        if agent.id in m.blocked_agents:
            return False
        if m.allowed_agents and agent.id not in m.allowed_agents:
            return False
        return True

    def search(self, task: str, agent: AgentSpec, token: ClientToken, limit: int = 3) -> List[Tuple[MemoryItem, float]]:
        allowed = [m for m in self.memory_items if self._allowed(m, agent, token)]
        if not allowed:
            return []
        docs = [m.title + " " + m.content for m in allowed]
        sims = tfidf_similarity(task, docs)
        scored = []
        for m, sim in zip(allowed, sims):
            score = (
                0.35 * float(sim)
                + 0.20 * 1.0
                + 0.15 * m.historical_usefulness
                + 0.10 * 0.75
                + 0.10 * m.authority
                - 0.20 * max(m.sensitivity - (1.0 - agent.risk_level), 0.0)
            )
            scored.append((m, max(score, 0.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


class AURARanker:
    def __init__(self, agents: Dict[str, AgentSpec], memory_store: MemoryStore):
        self.agents = agents
        self.memory_store = memory_store
        self.agent_rank = build_agent_rank()

    def rank_skills(self, task: str) -> pd.DataFrame:
        skill_ids = list(SKILL_DEFINITIONS.keys())
        docs = [SKILL_DEFINITIONS[s] + " " + " ".join(SKILL_KEYWORDS[s]) for s in skill_ids]
        sem = tfidf_similarity(task, docs)
        rows = []
        for skill_id, semantic in zip(skill_ids, sem):
            kw = contains_keyword(task, SKILL_KEYWORDS[skill_id])
            supply_quality = 0.0
            for a in self.agents.values():
                supply_quality += self.agent_rank.get(a.id, 0.0) * a.skill_proficiency.get(skill_id, 0.0)
            supply_quality = min(5.0 * supply_quality, 1.0)
            score = 0.45 * float(semantic) + 0.25 * kw + 0.15 * 0.5 + 0.15 * supply_quality
            rows.append({"skill_id": skill_id, "score": score, "semantic": float(semantic), "keyword": kw, "supply_quality": supply_quality})
        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    def semantic_agent_match(self, task: str, agent: AgentSpec) -> float:
        skill_text = " ".join([sid + " " + SKILL_DEFINITIONS[sid] for sid in agent.skill_proficiency])
        return float(tfidf_similarity(task, [agent.name + " " + agent.description + " " + skill_text])[0])

    def capability_match(self, agent: AgentSpec, skill_df: pd.DataFrame, top_k: int = 4) -> float:
        req = skill_df.head(top_k)[["skill_id", "score"]].values.tolist()
        total = sum(float(w) for _, w in req) or 1.0
        matched = sum(float(w) * agent.skill_proficiency.get(skill_id, 0.0) for skill_id, w in req)
        return min(matched / total, 1.0)

    def auth_readiness(self, agent: AgentSpec, required_auth: Set[str], connections: Dict[str, Connection]) -> float:
        needed = agent.required_auth | required_auth
        if not needed:
            return 1.0
        connected = {p for p, c in connections.items() if c.status == "connected"}
        return len(needed & connected) / len(needed)

    def latency_score(self, agent: AgentSpec, target_ms: int = 5000) -> float:
        return math.exp(-agent.p95_latency_ms / max(target_ms, 1))

    def cost_score(self, agent: AgentSpec, budget: float = 0.25) -> float:
        return 1.0 - min(agent.avg_cost_usd / max(budget, 1e-9), 1.0)

    def memory_fit(self, task: str, agent: AgentSpec, token: ClientToken) -> float:
        matches = self.memory_store.search(task, agent, token, limit=3)
        return matches[0][1] if matches else 0.0

    def risk_penalty(self, agent: AgentSpec, required_permissions: Set[str]) -> float:
        action_risk = 1.0 if (required_permissions & PermissionEngine.HIGH_RISK_PERMISSIONS) else 0.0
        return min(0.70 * agent.risk_level + 0.30 * action_risk, 1.0)

    def rank_agents(self, task: str, token: ClientToken, connections: Dict[str, Connection]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        skill_df = self.rank_skills(task)
        top_skills = skill_df.head(4)["skill_id"].tolist()
        required_permissions = PermissionEngine.required_permissions(top_skills)
        required_auth = PermissionEngine.required_auth(top_skills)
        rows = []
        for agent in self.agents.values():
            perm_fit = PermissionEngine.permission_fit(agent, token, required_permissions)
            auth_fit = self.auth_readiness(agent, required_auth, connections)
            missing_scope_penalty = 1.0 - perm_fit
            features = {
                "semantic_task_match": self.semantic_agent_match(task, agent),
                "capability_match": self.capability_match(agent, skill_df),
                "agent_rank": self.agent_rank.get(agent.id, 0.0),
                "user_history_success": agent.success_rate,
                "workspace_history_success": agent.success_rate,
                "auth_readiness": auth_fit,
                "permission_fit": perm_fit,
                "latency_score": self.latency_score(agent),
                "cost_score": self.cost_score(agent),
                "memory_fit": self.memory_fit(task, agent, token),
                "trust_score": agent.trust_score,
                "risk_penalty": self.risk_penalty(agent, required_permissions),
                "missing_scope_penalty": missing_scope_penalty,
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
            rows.append({
                "agent_id": agent.id,
                "agent_name": agent.name,
                "score": float(score),
                "top_skills": top_skills,
                "required_permissions": sorted(required_permissions),
                "required_auth": sorted(required_auth),
                **features,
            })
        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True), skill_df


class CostEstimator:
    @staticmethod
    def estimate_tokens(task: str, memory_items: List[MemoryItem], tools: List[str]) -> int:
        words = len(task.split())
        memory_words = sum(len(m.content.split()) for m in memory_items)
        tool_schema_tokens = 120 * len(tools)
        return int(words * 1.4 + memory_words * 1.4 + tool_schema_tokens + 700)

    @staticmethod
    def estimate_cost(agent: AgentSpec, tokens: int) -> float:
        # Use avg_cost as base; in real system use model pricing table.
        return round(agent.avg_cost_usd * max(tokens / 2500.0, 0.5), 6)


class AgentExecutor:
    def execute(self, agent: AgentSpec, task: str, memory_items: List[MemoryItem], approval_required: bool) -> ExecutionStep:
        tokens = CostEstimator.estimate_tokens(task, memory_items, list(agent.allowed_tools))
        cost = CostEstimator.estimate_cost(agent, tokens)
        latency = int(agent.p95_latency_ms * 0.65)
        if approval_required:
            output = f"Prepared action with {agent.name}, but waiting for human approval before high-risk execution."
            status = "waiting_for_approval"
        else:
            output = f"Mock result from {agent.name}: completed task using {len(memory_items)} memory items and {len(agent.allowed_tools)} tools."
            status = "completed"
        return ExecutionStep(
            step_id=str(uuid.uuid4()),
            agent_id=agent.id,
            agent_name=agent.name,
            input=task,
            output=output,
            tools_used=sorted(agent.allowed_tools),
            memory_used=[m.id for m in memory_items],
            tokens_estimated=tokens,
            cost_estimated=cost,
            latency_ms=latency,
            status=status,
            approval_required=approval_required,
        )


class AgentRouter:
    def __init__(self, ranker: AURARanker, memory_store: MemoryStore, executor: AgentExecutor):
        self.ranker = ranker
        self.memory_store = memory_store
        self.executor = executor

    def should_use_chain(self, task: str, skill_df: pd.DataFrame) -> bool:
        t = normalize_text(task)
        has_multi_intent = sum([
            any(x in t for x in ["research", "find", "search"]),
            any(x in t for x in ["book", "ticket", "reservation"]),
            any(x in t for x in ["calendar", "schedule", "event"]),
            any(x in t for x in ["write", "summary", "report"]),
        ]) >= 2
        return has_multi_intent

    def decompose(self, task: str) -> List[str]:
        t = normalize_text(task)
        subtasks = []
        if any(x in t for x in ["research", "find", "search", "travel", "flight", "hotel"]):
            subtasks.append("Find and compare options for: " + task)
        if any(x in t for x in ["book", "ticket", "reservation"]):
            subtasks.append("Prepare booking after approval for: " + task)
        if any(x in t for x in ["calendar", "schedule", "event"]):
            subtasks.append("Update calendar for: " + task)
        if any(x in t for x in ["write", "summary", "report"]):
            subtasks.append("Write final summary for: " + task)
        if len(subtasks) >= 2:
            subtasks.append("Review and validate the whole workflow for: " + task)
        return subtasks or [task]

    def run_task(self, task: str, token: ClientToken, connections: Dict[str, Connection]) -> ExecutionResult:
        start = time.time()
        ranking_df, skill_df = self.ranker.rank_agents(task, token, connections)
        use_chain = self.should_use_chain(task, skill_df)
        steps: List[ExecutionStep] = []
        selected_agent_ids: List[str] = []
        approval_required_any = False
        approval_reason = None

        if use_chain:
            subtasks = self.decompose(task)
            for subtask in subtasks:
                sub_rank_df, sub_skill_df = self.ranker.rank_agents(subtask, token, connections)
                best = sub_rank_df.iloc[0]
                agent = AGENTS[best["agent_id"]]
                req_perms = set(best["required_permissions"])
                approval_required, reason = PermissionEngine.approval_required(agent, req_perms)
                memory_matches = self.memory_store.search(subtask, agent, token, limit=3)
                memories = [m for m, _ in memory_matches]
                step = self.executor.execute(agent, subtask, memories, approval_required)
                steps.append(step)
                selected_agent_ids.append(agent.id)
                if approval_required:
                    approval_required_any = True
                    approval_reason = reason
        else:
            best = ranking_df.iloc[0]
            agent = AGENTS[best["agent_id"]]
            req_perms = set(best["required_permissions"])
            approval_required, reason = PermissionEngine.approval_required(agent, req_perms)
            memory_matches = self.memory_store.search(task, agent, token, limit=3)
            memories = [m for m, _ in memory_matches]
            step = self.executor.execute(agent, task, memories, approval_required)
            steps.append(step)
            selected_agent_ids.append(agent.id)
            approval_required_any = approval_required
            approval_reason = reason

        total_tokens = sum(s.tokens_estimated for s in steps)
        total_cost = round(sum(s.cost_estimated for s in steps), 6)
        total_latency = sum(s.latency_ms for s in steps)
        status = "waiting_for_approval" if approval_required_any else "completed"
        elapsed = int((time.time() - start) * 1000)

        return ExecutionResult(
            execution_id=str(uuid.uuid4()),
            workspace_id=token.workspace_id,
            user_id=token.user_id,
            client_name=token.client_name,
            task=task,
            selected_agent_ids=selected_agent_ids,
            status=status,
            approval_required=approval_required_any,
            approval_reason=approval_reason,
            total_tokens_estimated=total_tokens,
            total_cost_estimated=total_cost,
            total_latency_ms=max(total_latency, elapsed),
            steps=steps,
            ranking_table=ranking_df.head(6).to_dict(orient="records"),
        )


# ============================================================
# 5. Demo runner
# ============================================================

def result_to_pretty_dict(result: ExecutionResult) -> Dict[str, Any]:
    data = asdict(result)
    # Keep output readable.
    data["ranking_table"] = [
        {
            "agent_name": r["agent_name"],
            "score": round(r["score"], 4),
            "capability_match": round(r["capability_match"], 3),
            "permission_fit": round(r["permission_fit"], 3),
            "auth_readiness": round(r["auth_readiness"], 3),
            "risk_penalty": round(r["risk_penalty"], 3),
        }
        for r in result.ranking_table
    ]
    return data


def demo():
    memory_store = MemoryStore(MEMORY)
    ranker = AURARanker(AGENTS, memory_store)
    executor = AgentExecutor()
    router = AgentRouter(ranker, memory_store, executor)

    tasks = [
        "Summarize my unread Gmail and extract action items.",
        "Analyze open GitHub issues and create a triage summary.",
        "Give me a Stripe revenue report with MRR, ARR, churn and refunds.",
        "Book a flight from Hyderabad to Mumbai next Friday morning and add it to my calendar.",
        "Research competitors and write a short market brief with QA review.",
    ]

    for task in tasks:
        print("\n" + "=" * 100)
        print("TASK:", task)
        print("=" * 100)
        result = router.run_task(task, CLIENT_TOKEN, CONNECTIONS)
        pretty = result_to_pretty_dict(result)
        print("Status:", pretty["status"])
        print("Selected agents:", pretty["selected_agent_ids"])
        print("Approval required:", pretty["approval_required"], pretty["approval_reason"])
        print("Estimated tokens:", pretty["total_tokens_estimated"])
        print("Estimated cost:", pretty["total_cost_estimated"])
        print("\nTop ranking table:")
        print(pd.DataFrame(pretty["ranking_table"]).to_string(index=False))
        print("\nExecution steps:")
        for step in pretty["steps"]:
            print("-", step["agent_name"], "|", step["status"], "| tools:", step["tools_used"], "| memory:", step["memory_used"])
            print("  output:", step["output"])


if __name__ == "__main__":
    demo()
