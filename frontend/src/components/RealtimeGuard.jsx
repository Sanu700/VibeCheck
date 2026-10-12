import React, { useState, useEffect } from 'react';
import { Copy, RefreshCw, ArrowRight, ArrowDown, Check, Sparkles } from 'lucide-react';
import { getApiUrl } from '../config';

export default function RealtimeGuard() {
  const [text, setText] = useState('That is a stupid idea, you do not know anything!');
  const [selectedTone, setSelectedTone] = useState('Constructive');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const sampleMessages = [
    "That is a stupid idea, stop wasting our time.",
    "Hey team, great job on the release today!",
    "I am going to keep messaging you until you resign, loser.",
    "What the fuck is this code?",
    "I know you tried, but this implementation is unacceptable."
  ];

  const analyzeMessage = async (inputStr) => {
    if (!inputStr || !inputStr.trim()) {
      setAnalysis(null);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(getApiUrl('/api/analyze/message'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputStr,
          sender: 'You',
          requested_tones: ['Neutral', 'Friendly', 'Professional', 'Constructive']
        })
      });
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      console.error('Error analyzing message:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      analyzeMessage(text);
    }, 250);
    return () => clearTimeout(timer);
  }, [text]);

  const handleCopy = (str) => {
    navigator.clipboard.writeText(str);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderHeroToxicityHeader = (score) => {
    const pct = (score * 100).toFixed(1);
    const severity = analysis?.severity_level || (score >= 0.75 ? "High" : score >= 0.45 ? "Medium" : score >= 0.25 ? "Low" : "Clean");
    const isToxic = analysis?.is_toxic ?? (score >= 0.45);

    if (severity === "Critical" || severity === "High" || score >= 0.75) {
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <div className="flex items-baseline space-x-4">
              <span className="text-4xl sm:text-5xl font-extrabold font-mono tracking-tight text-red-400">{pct}%</span>
              <span className="text-sm font-bold px-3 py-1 rounded-md bg-red-950/80 border border-red-800 text-red-300">
                {severity === "Critical" ? "Critical Risk 🔴" : "High Risk 🔴"}
              </span>
            </div>
            <span className="text-sm text-zinc-300 font-medium">Detected Tone: <strong className="text-zinc-100 font-bold">{analysis.detected_tone}</strong></span>
          </div>
          <p className="text-sm text-zinc-300 pt-1">
            Your message contains highly aggressive, hostile, or inappropriate language.
          </p>
        </div>
      );
    } else if (severity === "Medium" || isToxic || score >= 0.45) {
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <div className="flex items-baseline space-x-4">
              <span className="text-4xl sm:text-5xl font-extrabold font-mono tracking-tight text-amber-400">{pct}%</span>
              <span className="text-sm font-bold px-3 py-1 rounded-md bg-amber-950/80 border border-amber-800 text-amber-300">
                Moderate ⚠
              </span>
            </div>
            <span className="text-sm text-zinc-300 font-medium">Detected Tone: <strong className="text-zinc-100 font-bold">{analysis.detected_tone}</strong></span>
          </div>
          <p className="text-sm text-zinc-300 pt-1">
            Your message may come across as hostile, sarcastic, or defensive.
          </p>
        </div>
      );
    } else if (severity === "Low" || score >= 0.25) {
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <div className="flex items-baseline space-x-4">
              <span className="text-4xl sm:text-5xl font-extrabold font-mono tracking-tight text-amber-300">{pct}%</span>
              <span className="text-sm font-bold px-3 py-1 rounded-md bg-amber-950/80 border border-amber-800 text-amber-200">
                Low Risk ⚠
              </span>
            </div>
            <span className="text-sm text-zinc-300 font-medium">Detected Tone: <strong className="text-zinc-100 font-bold">{analysis.detected_tone}</strong></span>
          </div>
          <p className="text-sm text-zinc-300 pt-1">
            Your message contains mild negativity or potential passive aggression.
          </p>
        </div>
      );
    } else {
      const safePct = (100 - parseFloat(pct)).toFixed(1);
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <div className="flex items-baseline space-x-4">
              <span className="text-4xl sm:text-5xl font-extrabold font-mono tracking-tight text-emerald-400">{safePct}%</span>
              <span className="text-sm font-bold px-3 py-1 rounded-md bg-emerald-950/80 border border-emerald-800 text-emerald-300">
                Safe ✓
              </span>
            </div>
            <span className="text-sm text-zinc-300 font-medium">Detected Tone: <strong className="text-zinc-100 font-bold">{analysis.detected_tone}</strong></span>
          </div>
          <p className="text-sm text-zinc-300 pt-1">
            Your message appears respectful and non-toxic (Toxicity score: {pct}%).
          </p>
        </div>
      );
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-10 py-4">
      
      {/* 1. Page Title */}
      <div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-zinc-100">Analyze a message</h1>
        <p className="text-base text-zinc-300 mt-1.5 font-normal">
          Check how your message may come across before sending it.
        </p>
      </div>

      {/* 2. Message Editor Workspace */}
      <div className="space-y-3">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden focus-within:border-orange-500/70 transition shadow-md">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            placeholder="Type your draft message here..."
            className="w-full bg-transparent p-4 text-zinc-100 placeholder-zinc-500 focus:outline-none text-base leading-relaxed font-sans"
          />

          <div className="flex items-center justify-between px-4 py-3 bg-zinc-950/90 border-t border-zinc-800">
            <span className="text-xs font-mono text-zinc-400">{text.length} characters</span>
            
            <button
              onClick={() => analyzeMessage(text)}
              disabled={loading || !text.trim()}
              className="bg-orange-500 hover:bg-orange-600 text-zinc-950 font-bold text-sm px-5 py-2 rounded-lg transition flex items-center gap-2 shadow-sm disabled:opacity-50"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : null}
              <span>Analyze message</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Sample Links */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs text-zinc-400">
          <span className="text-zinc-500 font-medium">Try sample:</span>
          {sampleMessages.map((sample, idx) => (
            <button
              key={idx}
              onClick={() => setText(sample)}
              className="text-zinc-300 hover:text-orange-400 underline underline-offset-2 transition"
            >
              Sample {idx + 1}
            </button>
          ))}
        </div>
      </div>

      {/* 3. HERO Analysis Section */}
      {analysis && (
        <div className="space-y-8 pt-6 border-t border-zinc-800">
          
          {/* HERO Toxicity Header & Category Breakdown */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-6 shadow-md">
            
            <div className="border-b border-zinc-800 pb-5">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-400 block mb-2">Toxicity Result</span>
              {renderHeroToxicityHeader(analysis.overall_score)}
            </div>

            {/* Compact Ranked Category List */}
            <div className="space-y-3">
              <span className="text-xs font-mono font-semibold uppercase tracking-wider text-zinc-400 block">Multi-Category Scores</span>
              
              <div className="space-y-2.5">
                {Object.entries(analysis.categories)
                  .sort(([, a], [, b]) => b - a)
                  .map(([cat, score]) => {
                    const pct = Math.round(score * 100);
                    
                    let textColor = "text-zinc-300";
                    let barColor = "bg-zinc-700";
                    if (pct >= 50) {
                      textColor = "text-red-400 font-bold";
                      barColor = "bg-red-500";
                    } else if (pct >= 25) {
                      textColor = "text-amber-400 font-semibold";
                      barColor = "bg-amber-500";
                    }

                    return (
                      <div key={cat} className="flex items-center text-sm gap-4 font-mono">
                        <span className={`w-36 capitalize font-sans ${textColor}`}>
                          {cat.replace('_', ' ')}
                        </span>
                        <div className="flex-1 bg-zinc-950 rounded-full h-2 overflow-hidden border border-zinc-800">
                          <div
                            className={`h-full ${barColor} transition-all duration-300`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className={`w-12 text-right ${textColor}`}>{pct}%</span>
                      </div>
                    );
                  })}
              </div>
            </div>

          </div>

          {/* 4. CONSTRUCTIVE REWRITE WORKFLOW */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 space-y-6 shadow-md">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-4">
              <div>
                <h3 className="text-lg font-bold text-zinc-100 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-orange-400" /> VibeCheck Suggests
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">Reframing message intent into constructive language.</p>
              </div>

              {/* Tone Selection Tabs */}
              <div className="flex bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-medium">
                {['Constructive', 'Professional', 'Friendly', 'Neutral'].map((tone) => (
                  <button
                    key={tone}
                    onClick={() => setSelectedTone(tone)}
                    className={`px-3 py-1.5 rounded-md transition ${
                      selectedTone === tone
                        ? 'bg-zinc-800 text-zinc-100 font-bold border border-zinc-700 shadow-sm'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {tone}
                  </button>
                ))}
              </div>
            </div>

            {/* Original vs Suggested Transformation Flow */}
            <div className="space-y-4">
              <div className="space-y-1.5">
                <span className="text-xs font-mono font-semibold text-zinc-400 uppercase">Your original message</span>
                <div className="bg-zinc-950 p-3.5 rounded-lg text-sm text-zinc-300 border border-zinc-800 font-mono">
                  "{text}"
                </div>
              </div>

              <div className="flex justify-center text-zinc-500">
                <ArrowDown className="w-5 h-5 text-orange-400" />
              </div>

              <div className="space-y-1.5">
                <span className="text-xs font-mono font-semibold text-orange-400 uppercase">Suggested rewrite ({selectedTone})</span>
                <div className="bg-zinc-950 border border-orange-500/30 p-4 rounded-lg text-base text-zinc-100 font-medium leading-relaxed">
                  "{analysis.rewrites[selectedTone] || text}"
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-end space-x-3 pt-2 border-t border-zinc-800">
              <button
                onClick={() => handleCopy(analysis.rewrites[selectedTone])}
                className="px-4 py-2 rounded-lg text-xs font-semibold text-zinc-200 border border-zinc-700 hover:bg-zinc-800 transition flex items-center gap-1.5"
              >
                <Copy className="w-3.5 h-3.5" />
                <span>{copied ? 'Copied to clipboard' : 'Copy rewrite'}</span>
              </button>
              <button
                onClick={() => setText(analysis.rewrites[selectedTone])}
                className="px-4 py-2 rounded-lg text-xs font-bold bg-orange-500 hover:bg-orange-600 text-zinc-950 transition flex items-center gap-1.5 shadow-sm"
              >
                <Check className="w-4 h-4 text-zinc-950 stroke-[3]" />
                <span>Replace message</span>
              </button>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}
