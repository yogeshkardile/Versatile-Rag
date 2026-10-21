import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

type Method = "bm25" | "dense" | "hybrid" | "rerank";
type Doc = {id: string; filename: string; pages: number; chunks: number; size_bytes?: number};
type Source = {chunk_id: string; filename: string; page: number; text: string; rank: number; score: number};
type PerformanceRow = {
  method: string;
  mean_latency_ms: number | null;
  requests?: number;
  last_latency_ms?: number | null;
  doc_recall_at_10?: number;
  section_recall_at_10?: number;
  section_mrr_at_10?: number;
};
type Capabilities = {
  generation: {available: boolean; mode: string};
  dense: {available: boolean};
  reranker: {available: boolean};
};

const methodLabels: Record<Method, string> = {
  bm25: "BM25",
  dense: "Dense retrieval",
  hybrid: "Hybrid RRF",
  rerank: "Hybrid + reranking",
};

const methodDescriptions: Record<Method, string> = {
  bm25: "Fast lexical matching for exact terms and phrases.",
  dense: "Semantic matching with the BGE embedding model.",
  hybrid: "Combines lexical and semantic rankings.",
  rerank: "Refines hybrid results with a cross-encoder.",
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : "The backend request failed.");
  }
  return payload as T;
}

function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState<Method>("hybrid");
  const [answer, setAnswer] = useState("");
  const [answerMode, setAnswerMode] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [historical, setHistorical] = useState<PerformanceRow[]>([]);
  const [current, setCurrent] = useState<PerformanceRow[]>([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [documents, performance, caps] = await Promise.all([
        api<Doc[]>("/documents"),
        api<{historical_phase5_2: PerformanceRow[]; current_session: PerformanceRow[]}>("/performance"),
        api<Capabilities>("/capabilities"),
      ]);
      setDocs(documents);
      setHistorical(performance.historical_phase5_2);
      setCurrent(performance.current_session);
      setCapabilities(caps);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Backend unavailable.");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const upload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    setError("");
    setStatus("Uploading and indexing your PDF...");
    try {
      const body = new FormData();
      body.append("file", file);
      const data = await api<Doc>("/documents", {method: "POST", body});
      setStatus(`Ready: ${data.filename} was indexed successfully.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
      setStatus("");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (doc: Doc) => {
    if (!window.confirm(`Delete "${doc.filename}" and all indexed evidence?`)) return;
    setBusy(true);
    setError("");
    try {
      await api(`/documents/${doc.id}`, {method: "DELETE"});
      setStatus(`Removed ${doc.filename} from the workspace.`);
      setSources([]);
      setAnswer("");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Deletion failed.");
    } finally {
      setBusy(false);
    }
  };

  const ask = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setError("");
    setStatus("Searching your indexed evidence...");
    try {
      const data = await api<{
        answer: string;
        answer_mode: string;
        generation_status: string;
        generation_model: string | null;
        generation_latency_ms: number;
        generation_error: string | null;
        sources: Source[];
        latency_ms: number;
      }>("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question, method, limit: 5}),
      });
      setAnswer(data.answer);
      setAnswerMode(data.answer_mode);
      setSources(data.sources);
      if (data.generation_error) {
        setError(`Generation failed; showing extractive fallback: ${data.generation_error}`);
      }
      setStatus(`${methodLabels[method]} completed in ${data.latency_ms.toFixed(1)} ms.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Question failed.");
      setStatus("");
    } finally {
      setBusy(false);
    }
  };

  const clearResults = () => {
    setAnswer("");
    setAnswerMode("");
    setSources([]);
    setStatus("");
    setError("");
  };

  const selectedCapability = method === "bm25"
    ? true
    : method === "rerank"
      ? capabilities?.reranker.available
      : capabilities?.dense.available;
  const historicalMap = useMemo(() => new Map(historical.map((row) => [row.method, row])), [historical]);
  const currentRequests = current.reduce((total, row) => total + (row.requests ?? 0), 0);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Versatile RAG home">
          <span className="brand-mark">V</span>
          <span><strong>Versatile RAG</strong><small>Scientific evidence workspace</small></span>
        </a>
        <nav className="topnav" aria-label="Workspace sections">
          <a className="active" href="#workspace">Workspace</a>
          <a href="#performance">Performance</a>
        </nav>
        <span className="connection-pill"><span className="connection-dot" />Private workspace</span>
      </header>

      <section className="welcome" id="top">
        <div>
          <span className="eyebrow">RESEARCH ASSISTANT</span>
          <h1>Ask better questions of your papers.</h1>
          <p>Upload scientific PDFs, retrieve supporting passages, and review page-level evidence in one focused workspace.</p>
        </div>
        <button className="primary-button upload-button" onClick={() => fileInput.current?.click()} disabled={busy}>
          <span className="button-icon">+</span> Upload PDF
          <input ref={fileInput} type="file" accept="application/pdf" onChange={upload} disabled={busy} />
        </button>
      </section>

      {(error || status) && (
        <div className={`notice ${error ? "notice-error" : "notice-success"}`} role={error ? "alert" : "status"}>
          <span className="notice-icon">{error ? "!" : "✓"}</span>
          <span>{error || status}</span>
        </div>
      )}

      <section className="summary-grid" aria-label="Workspace summary">
        <div className="summary-card"><span className="summary-icon sage">▤</span><span><small>Indexed documents</small><strong>{docs.length}</strong></span></div>
        <div className="summary-card"><span className="summary-icon blue">⌁</span><span><small>Retrieval method</small><strong>{methodLabels[method]}</strong></span></div>
        <div className="summary-card"><span className="summary-icon gold">◌</span><span><small>Current requests</small><strong>{currentRequests}</strong></span></div>
        <div className="summary-card"><span className={`summary-icon ${capabilities?.generation.available ? "sage" : "muted"}`}>✦</span><span><small>Answer provider</small><strong>{capabilities?.generation.available ? "Available" : "Fallback mode"}</strong></span></div>
      </section>

      <section className="content-grid" id="workspace">
        <aside className="panel library-panel">
          <div className="section-header">
            <div><span className="section-label">YOUR LIBRARY</span><h2>Documents</h2></div>
            <span className="count-badge">{docs.length}</span>
          </div>
          <button className="secondary-button full-width" onClick={() => fileInput.current?.click()} disabled={busy}>+ Add another PDF</button>
          {docs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-illustration">PDF</div>
              <strong>Your library is empty</strong>
              <span>Upload a PDF to build a private searchable collection.</span>
            </div>
          ) : (
            <div className="document-list">
              {docs.map((doc) => (
                <div className="document-card" key={doc.id}>
                  <div className="document-icon">PDF</div>
                  <div className="document-info"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.pages} pages · {doc.chunks} chunks</small></div>
                  <button className="icon-button" title={`Delete ${doc.filename}`} onClick={() => void remove(doc)} disabled={busy}>×</button>
                </div>
              ))}
            </div>
          )}
          <div className="privacy-note"><span className="lock-icon">⌑</span><div><strong>Private by design</strong><span>Your files stay in local application storage and are never publicly shared.</span></div></div>
        </aside>

        <section className="panel question-panel">
          <div className="section-header question-header">
            <div><span className="section-label">QUESTION WORKSPACE</span><h2>Ask your documents</h2></div>
            {answer && <button className="text-button" onClick={clearResults}>Clear results</button>}
          </div>
          <form onSubmit={ask}>
            <label className="field-label" htmlFor="question">What would you like to understand?</label>
            <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="e.g. What are the main findings of this paper?" aria-label="Question" disabled={busy} />
            <div className="method-row">
              <div className="method-select-wrap">
                <label className="field-label" htmlFor="method">Retrieval strategy</label>
                <select id="method" value={method} onChange={(event) => setMethod(event.target.value as Method)} disabled={busy}>
                  {Object.entries(methodLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}
                </select>
              </div>
              <div className="method-description"><span className="info-dot">i</span><span>{methodDescriptions[method]}</span></div>
              <button className="primary-button ask-button" disabled={!question.trim() || busy}>{busy ? <><span className="spinner" /> Working...</> : <>Retrieve evidence <span>→</span></>}</button>
            </div>
          </form>
          {method !== "bm25" && capabilities && !selectedCapability && <div className="model-note"><strong>Model not ready</strong><span>This method requires a local model. It will report a clear error if unavailable and will not silently switch methods.</span></div>}

          {answer ? (
            <article className="answer-card">
              <div className="result-heading"><div><span className="section-label">RESPONSE</span><h3>Evidence-grounded response</h3></div><span className="badge">{answerMode === "extractive_context_fallback" ? "Extractive fallback" : answerMode}</span></div>
              <p>{answer}</p>
              {answerMode === "extractive_context_fallback" && <small>This is retrieved context, not a model-generated answer. Configure a generation provider for generated responses.</small>}
            </article>
          ) : (
            <div className="answer-placeholder"><span className="placeholder-icon">✦</span><div><strong>Your response will appear here</strong><span>Ask a question to retrieve evidence from your indexed PDFs.</span></div></div>
          )}

          <div className="sources-heading"><div><span className="section-label">EVIDENCE</span><h3>Retrieved sources</h3></div>{sources.length > 0 && <span className="source-count">{sources.length} passages</span>}</div>
          {sources.length === 0 ? <div className="empty-sources">Source snippets, page numbers, ranks, and scores will appear after retrieval.</div> :
            <div className="source-list">{sources.map((source) => (
              <article className="source-card" key={source.chunk_id}>
                <div className="source-number">{source.rank}</div>
                <div className="source-body"><div className="source-meta"><strong>{source.filename}</strong><span>Page {source.page} <i /> Score {source.score.toFixed(4)}</span></div><p>{source.text}</p></div>
              </article>
            ))}</div>}
        </section>
      </section>

      <section className="panel performance-panel" id="performance">
        <div className="section-header"><div><span className="section-label">OBSERVABILITY</span><h2>Performance overview</h2></div><span className="history-tag">Historical + live</span></div>
        <p className="panel-description">Compare the validated Phase 5.2 research baseline with measurements from this application session. They should not be interpreted as the same experiment.</p>
        <div className="table-wrap"><table><thead><tr><th>Method</th><th>Phase 5.2 mean</th><th>Current mean</th><th>Requests</th><th>Research section R@10</th></tr></thead><tbody>
          {Object.values(methodLabels).map((label) => {
            const old = historicalMap.get(label);
            const live = current.find((row) => row.method === label);
            return <tr key={label}><td><span className="method-dot" /><strong>{label}</strong></td><td>{old ? `${old.mean_latency_ms?.toFixed(2)} ms` : "—"}</td><td>{live?.mean_latency_ms == null ? "—" : `${live.mean_latency_ms.toFixed(2)} ms`}</td><td>{live?.requests ?? 0}</td><td>{old?.section_recall_at_10 == null ? "—" : old.section_recall_at_10.toFixed(4)}</td></tr>;
          })}
        </tbody></table></div>
      </section>

      <footer><span>Versatile RAG · Local scientific document workspace</span><span>Models load lazily and remain optional.</span></footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
