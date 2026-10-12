import React from 'react';
import { Shield, RefreshCw, AlertCircle, Server } from 'lucide-react';

export default function BackendReadiness({
  status,
  elapsedSeconds = 0,
  attempt = 1,
  onRetry,
  targetUrl = ''
}) {
  const isUnavailable = status === 'unavailable';

  // Dynamic friendly copy that evolves without fake timers or percentage promises
  const getHeadline = () => {
    if (isUnavailable) return 'Backend service unreachable';
    if (elapsedSeconds < 4) return 'Connecting to VibeCheck…';
    if (elapsedSeconds < 18) return 'Waking up the vibe checker…';
    return 'Almost there, preparing services…';
  };

  const getSubhead = () => {
    if (isUnavailable) {
      return 'The backend server did not respond within the expected time window. Free instances may still be waking up, or the network connection may be interrupted.';
    }
    if (elapsedSeconds < 4) {
      return 'Verifying API status and moderation pipeline readiness.';
    }
    if (elapsedSeconds < 18) {
      return 'Free cloud instances pause during periods of inactivity and take a brief moment to spin back up.';
    }
    return 'The server is completing boot sequences. Your session will connect automatically the moment it is ready.';
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col justify-between selection:bg-orange-500/20 selection:text-orange-200">
      {/* Header bar matching main Navbar structure */}
      <header className="border-b border-zinc-800/80 bg-zinc-950/95 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-orange-500 flex items-center justify-center text-zinc-950 font-bold shadow-sm">
              <Shield className="w-4 h-4 text-zinc-950 fill-zinc-950" />
            </div>
            <span className="font-bold text-zinc-100 text-lg tracking-tight">VibeCheck</span>
          </div>

          <div className="flex items-center space-x-2 text-xs text-zinc-400 border border-zinc-800 bg-zinc-900/80 px-3 py-1.5 rounded-md font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                isUnavailable
                  ? 'bg-rose-500'
                  : 'bg-amber-400 animate-pulse'
              }`}
            />
            <span>{isUnavailable ? 'Offline' : 'Connecting'}</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-lg w-full mx-auto px-4 flex flex-col items-center justify-center text-center py-12">
        <div className="w-full flex flex-col items-center">
          
          {/* Status Badge / Icon */}
          <div className="mb-6 relative flex items-center justify-center">
            {isUnavailable ? (
              <div className="w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
                <AlertCircle className="w-8 h-8" />
              </div>
            ) : (
              <div className="relative flex items-center justify-center">
                {/* Subtle outer breathing ring */}
                <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center shadow-inner">
                  <Shield className="w-7 h-7 text-orange-500 transition-transform" />
                </div>
                {/* Subtle minimal spinner around the container */}
                <div className="absolute -inset-1.5 rounded-[1.25rem] border border-orange-500/30 border-t-orange-500 animate-spin" />
              </div>
            )}
          </div>

          {/* Typography */}
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-zinc-100 mb-3">
            {getHeadline()}
          </h1>
          <p className="text-sm text-zinc-400 max-w-md leading-relaxed mb-6">
            {getSubhead()}
          </p>

          {/* Dynamic details pill */}
          {!isUnavailable ? (
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-ping" />
              <span>Probe attempt {attempt}</span>
              {elapsedSeconds > 0 && <span className="text-zinc-500">• {elapsedSeconds}s</span>}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4 w-full">
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-zinc-950 font-semibold text-sm transition-all duration-150 shadow-sm active:scale-95 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 focus:ring-offset-zinc-950"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Retry Connection</span>
              </button>

              {targetUrl && (
                <div className="inline-flex items-center gap-1.5 text-[11px] text-zinc-500 font-mono bg-zinc-900/50 px-2.5 py-1 rounded border border-zinc-800/60 max-w-sm truncate">
                  <Server className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate">Target: {targetUrl}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Footer matching standard app footer */}
      <footer className="border-t border-zinc-900 bg-zinc-950 py-5 text-zinc-500 text-xs">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2 font-mono">
          <span>VibeCheck • Conversational Vibe Analysis & Moderation Engine</span>
          <span>FastAPI • React • Vite • Scikit-Learn</span>
        </div>
      </footer>
    </div>
  );
}
