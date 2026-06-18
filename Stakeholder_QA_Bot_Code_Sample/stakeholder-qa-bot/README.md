# Stakeholder Q&A Bot — Code Sample

A natural-language analytics agent: users ask questions in plain English, and a **LangGraph** supervisor routes them through **text-to-SQL** and **semantic RAG** paths, then synthesizes a grounded answer. This is the architecture behind the stakeholder-facing dashboard Q&A bot.

> **Note on this sample.** This is a self-contained extract of a larger personal project (an automated job-search assistant), included to show how I architect an agentic text-to-SQL + RAG system. The data domain here is job postings rather than the healthcare/network domain of the main presentation, but the engineering — graph orchestration, routing, governed SQL, retrieval, reflection, and evaluation — is the same pattern described for the stakeholder bot. It is not wired to any production data.

## Why this is worth reading

- **Explicit state-machine design.** `agent/graph.py` is a LangGraph supervisor composing three subgraphs (retrieval, analytics, reasoning); `agent/state.py` is a typed `AgentState` with per-field reducers (`add` vs keep-last) — control flow is engineered, not improvised by the model.
- **Single-responsibility nodes.** `agent/nodes/` — router, sql_node, rag_node, term_expander, synthesizer, reflector, followup, answer — each small and individually testable.
- **Governed text-to-SQL.** `retrieval/sql.py` generates SQL against a fixed schema with a one-retry-on-error loop; the SQL path is the only effector.
- **Real retrieval engineering.** `retrieval/rag.py` does Qdrant hybrid search (dense + sparse, RRF fusion) with a cross-encoder reranker.
- **A real evaluation suite.** `eval/` has metric definitions (sql_correctness, completeness, actionability, LLM-as-judge) and pytest-based routing/SQL/agent tests — testing an agentic system, not just shipping it.

## Layout

```
agent/
  graph.py            LangGraph supervisor: routes into subgraphs
  state.py            Typed shared state with reducers
  prompts.py          Node prompts
  llm_client.py       Backend-agnostic LLM client
  nodes/              router · sql_node · rag_node · term_expander
                      synthesizer · reflector · followup · answer · summarizer
  subgraphs/          retrieval · analytics · reasoning
retrieval/
  sql.py              Text-to-SQL generation + execution
  rag.py              Hybrid dense+sparse retrieval + reranking
ingest/               Chunking, embedding, Qdrant vector store, record management
cache/                Query cache
eval/                 Metrics, datasets, and pytest eval suites
cli.py                Entry point (ingest | ask | eval | status)
config.py
```

## Start here
- `PILOT_INTEL.md` — full technical architecture (module map, data flow, design rationale).
- `CONTEXT.md` — the runtime context/schema description injected into nodes.
- `agent/graph.py` → `agent/state.py` → `agent/nodes/sql_node.py` → `retrieval/sql.py` is the shortest path through the core idea.

*Original project license: AGPL-3.0 (carried over from the source repository).*
