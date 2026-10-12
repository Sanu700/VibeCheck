import React, { useState, useEffect, useRef, useCallback } from 'react';
import Navbar from './components/Navbar';
import RealtimeGuard from './components/RealtimeGuard';
import ConversationAnalyzer from './components/ConversationAnalyzer';
import InsightsView from './components/InsightsView';
import BackendReadiness from './components/BackendReadiness';
import { getApiUrl, API_BASE_URL } from './config';

const MAX_WAIT_SECONDS = 60;
const REQUEST_TIMEOUT_MS = 6000;
const RETRY_DELAY_MS = 2500;

export default function App() {
  const [mainTab, setMainTab] = useState('analyze');
  const [isReady, setIsReady] = useState(false);
  const [readinessStatus, setReadinessStatus] = useState('probing');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [attempt, setAttempt] = useState(1);
  const [apiStatus, setApiStatus] = useState('checking');

  const abortControllerRef = useRef(null);
  const isCancelledRef = useRef(false);
  const timerRef = useRef(null);
  const elapsedTimerRef = useRef(null);

  const performHealthCheck = async (signal) => {
    // Try primary health endpoint (/api/health)
    try {
      const res = await fetch(getApiUrl('/api/health'), {
        signal,
        cache: 'no-store'
      });
      if (res.ok) {
        return true;
      }
    } catch (e) {
      if (signal?.aborted) throw e;
    }

    // Fallback to top-level health endpoint (/health)
    try {
      const res = await fetch(getApiUrl('/health'), {
        signal,
        cache: 'no-store'
      });
      if (res.ok) {
        return true;
      }
    } catch (e) {
      if (signal?.aborted) throw e;
    }

    return false;
  };

  const startProbing = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
    }

    isCancelledRef.current = false;
    setReadinessStatus('probing');
    setElapsedSeconds(0);
    setAttempt(1);

    const startTime = Date.now();
    elapsedTimerRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setElapsedSeconds(elapsed);
      if (elapsed >= MAX_WAIT_SECONDS) {
        if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
        if (abortControllerRef.current) abortControllerRef.current.abort();
        setReadinessStatus('unavailable');
        setApiStatus('offline');
      }
    }, 1000);

    let currentAttempt = 1;

    const probeCycle = async () => {
      if (isCancelledRef.current) return;

      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      if (elapsed >= MAX_WAIT_SECONDS) {
        setReadinessStatus('unavailable');
        setApiStatus('offline');
        if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
        return;
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const timeoutId = setTimeout(() => {
        controller.abort();
      }, REQUEST_TIMEOUT_MS);

      let success = false;
      try {
        success = await performHealthCheck(controller.signal);
      } catch (err) {
        success = false;
      } finally {
        clearTimeout(timeoutId);
      }

      if (isCancelledRef.current) return;

      if (success) {
        if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
        setApiStatus('online');
        setIsReady(true);
        return;
      }

      currentAttempt += 1;
      setAttempt(currentAttempt);

      const updatedElapsed = Math.floor((Date.now() - startTime) / 1000);
      if (updatedElapsed < MAX_WAIT_SECONDS && !isCancelledRef.current) {
        timerRef.current = setTimeout(probeCycle, RETRY_DELAY_MS);
      } else {
        if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
        setReadinessStatus('unavailable');
        setApiStatus('offline');
      }
    };

    probeCycle();
  }, []);

  useEffect(() => {
    startProbing();

    return () => {
      isCancelledRef.current = true;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
      }
    };
  }, [startProbing]);

  // Periodic lightweight background health monitor when ready
  useEffect(() => {
    if (!isReady) return;

    const interval = setInterval(async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const ok = await performHealthCheck(controller.signal);
        clearTimeout(timeoutId);
        setApiStatus(ok ? 'online' : 'degraded');
      } catch (e) {
        setApiStatus('degraded');
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [isReady]);

  if (!isReady) {
    return (
      <BackendReadiness
        status={readinessStatus}
        elapsedSeconds={elapsedSeconds}
        attempt={attempt}
        onRetry={startProbing}
        targetUrl={API_BASE_URL}
      />
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans animate-fadeIn">
      {/* Top Navbar */}
      <Navbar mainTab={mainTab} setMainTab={setMainTab} apiStatus={apiStatus} />

      {/* Main Container Workspace */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-8">
        {mainTab === 'analyze' && <RealtimeGuard />}
        {mainTab === 'conversations' && <ConversationAnalyzer />}
        {mainTab === 'insights' && <InsightsView />}
      </main>

      {/* Subtle Minimal Footer */}
      <footer className="border-t border-zinc-900 bg-zinc-950 py-5 text-zinc-500 text-xs">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2 font-mono">
          <span>VibeCheck • Conversational Vibe Analysis & Moderation Engine</span>
          <span>FastAPI • React • Vite • Scikit-Learn</span>
        </div>
      </footer>
    </div>
  );
}
