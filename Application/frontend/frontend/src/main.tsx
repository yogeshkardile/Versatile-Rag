import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

type Method = "bm25" | "dense" | "hybrid" | "rerank";
type Doc = {id: string; filename: string; pages: number; chunks: number; size_bytes?: number};
type Source = {chunk_id: string; filename: string; page: number; text: string; rank: number; score: number};
type PerformanceRow = {method: string; mean_latency_ms: number | null; requests?: number; last_latency_ms?: number | null; doc_recall_at_10?: number; section_recall_at_10?: number; section_mrr_at_10?: number};

const methodLabels: Record<Method, string> = {
  bm25: "BM25",
  dense: "Dense retrieval",
  hybrid: "Hybrid RRF",
  rerank: "Hybrid + reranking",
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "The backend request failed.");
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
  const [capabilities, setCapabilities] = useState<{generation: {available: boolean; mode: string}; dense: {available: boolean}; reranker: {available: boolean}} | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [documents, performance, caps] = await Promise.all([
        api<Doc[]>("/documents"),
        api<{historical_phase5_2: PerformanceRow[]; current_session: PerformanceRow[]}>("/performance"),
        api<typeof capabilities>("/capabilities"),
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
    setBusy(true); setError(""); setStatus("Processing PDF...");
    try {
      const body = new FormData(); body.append("file", file);
      const data = await api<Doc>("/documents", {method: "POST", body});
      setStatus(`Indexed ${data.filename}: ${data.pages} page${data.pages === 1 ? "" : "s"}.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
      setStatus("");
    } finally { setBusy(false); }
  };

  const remove = async (doc: Doc) => {
    if (!window.confirm(`Delete "${doc.filename}" and all indexed evidence?`)) return;
    setBusy(true); setError("");
    try {
      await api(`/documents/${doc.id}`, {method: "DELETE"});
      setStatus(`Deleted ${doc.filename}.`);
      setSources([]); setAnswer("");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Deletion failed.");
    } finally { setBusy(false); }
  };

  const ask = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true); setError(""); setStatus("Retrieving evidence...");
    try {
      const data = await api<{answer: string; answer_mode: string; generation_status: string; generation_model: string | null; generation_latency_ms: number; generation_error: string | null; sources: Source[]; latency_ms: number}>("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question, method, limit: 5}),
      });
      setAnswer(data.answer); setAnswerMode(data.answer_mode);
      setSources(data.sources);
      if (data.generation_error) setError(`Generation failed; showing extractive fallback: ${data.generation_error}`);
      setStatus(`${methodLabels[method]} completed in ${data.latency_ms.toFixed(1)} ms.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Question failed.");
      setStatus("");
    } finally { setBusy(false); }
  };

  const selectedCapability = method === "bm25" ? true : method === "rerank" ? capabilities?.reranker.available : capabilities?.dense.available;
  const historicalMap = useMemo(() => new Map(historical.map((row) => [row.method, row])), [historical]);

  return <main>
    <header className="hero">
      <div>
        <span className="eyebrow">VERSATILE RAG · MVP</span>
        <h1>Scientific document workspace</h1>
        <p>Upload private PDFs, ask evidence-seeking questions, and inspect exactly which pages support the response.</p>
      </div>
      <label className="upload-button">Upload PDF<input ref={fileInput} type="file" accept="application/pdf" onChange={upload} disabled={busy}/></label>
    </header>
    {error && <div className="alert error" role="alert">{error}</div>}
    {status && <div className="alert status">{status}</div>}
    <section className="workspace-grid">
      <aside className="panel documents-panel">
        <div className="panel-heading"><div><span className="section-label">LIBRARY</span><h2>Documents</h2></div><span className="count">{docs.length}</span></div>
        {docs.length === 0 ? <div className="empty-state"><strong>No PDFs indexed</strong><span>Upload a scientific PDF to start a private workspace.</span></div> :
          <div className="document-list">{docs.map((doc) => <div className="document-card" key={doc.id}>
            <div className="document-icon">PDF</div><div className="document-info"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.pages} pages · {doc.chunks} chunks</small></div>
            <button className="icon-button" title={`Delete ${doc.filename}`} onClick={() => void remove(doc)} disabled={busy}>×</button>
          </div>)}</div>}
        <div className="privacy-note"><strong>Private workspace</strong><span>Files stay in the local backend storage and are not publicly shared.</span></div>
      </aside>
      <section className="panel qa-panel">
        <div className="panel-heading"><div><span className="section-label">EVIDENCE WORKSPACE</span><h2>Ask a question</h2></div>
          <select aria-label="Retrieval method" value={method} onChange={(event) => setMethod(event.target.value as Method)} disabled={busy}>
            {Object.entries(methodLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}
          </select>
        </div>
        {method !== "bm25" && capabilities && !selectedCapability && <div className="model-note">This method needs a local model that has not loaded yet. The first request may download or initialize it; failures are reported without falling back silently.</div>}
        <form onSubmit={ask}>
          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What does the paper conclude about..." aria-label="Question" disabled={busy}/>
          <div className="form-footer"><span className="hint">Retrieval is text-only and source pages are shown below.</span><button className="primary-button" disabled={!question.trim() || busy}>{busy ? "Working..." : "Retrieve evidence"}</button></div>
        </form>
        {answer && <article className="answer-card"><div className="answer-heading"><h3>Response</h3><span className="badge">{answerMode === "extractive_context_fallback" ? "Extractive fallback" : answerMode}</span></div><p>{answer}</p>{answerMode === "extractive_context_fallback" && <small>No generation provider is configured. This is the top retrieved context, not a generated answer.</small>}</article>}
        <div className="sources-heading"><h3>Sources</h3>{sources.length > 0 && <span>{sources.length} retrieved</span>}</div>
        {sources.length === 0 ? <div className="empty-sources">Retrieved snippets and page references will appear here.</div> :
          <div>{sources.map((source) => <article className="source-card" key={source.chunk_id}><div className="source-meta"><strong>{source.filename}</strong><span>Page {source.page} · Rank {source.rank} · Score {source.score.toFixed(4)}</span></div><p>{source.text}</p></article>)}</div>}
      </section>
    </section>
    <section className="panel performance-panel">
      <div className="panel-heading"><div><span className="section-label">MEASUREMENTS</span><h2>Retrieval performance</h2></div><span className="history-tag">Historical vs current</span></div>
      <p className="panel-description">Historical values are the validated Phase 5.2 research comparison. Current values are only this browser session and are not research results.</p>
      <div className="table-wrap"><table><thead><tr><th>Method</th><th>Phase 5.2 mean</th><th>Current mean</th><th>Current requests</th><th>Research section R@10</th></tr></thead><tbody>
        {Object.values(methodLabels).map((label) => { const old = historicalMap.get(label); const live = current.find((row) => row.method === label); return <tr key={label}><td><strong>{label}</strong></td><td>{old ? `${old.mean_latency_ms?.toFixed(2)} ms` : "—"}</td><td>{live?.mean_latency_ms == null ? "—" : `${live.mean_latency_ms.toFixed(2)} ms`}</td><td>{live?.requests ?? 0}</td><td>{old?.section_recall_at_10 == null ? "—" : old.section_recall_at_10.toFixed(4)}</td></tr>; })}
      </tbody></table></div>
    </section>
    <footer><span>Models load lazily and remain optional.</span><span>Phase 5.2 metrics apply to its sampled corpus and queries only.</span></footer>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
