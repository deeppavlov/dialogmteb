import { Languages, ListChecks, Globe } from "lucide-react";

interface TaskCardProps {
  title: string;
  stats: { languages: string; tasks: string; domains: string };
  datasets: string[];
  taskType: "classification" | "retrieval" | "reranking" | "pair" | "sts";
}

export function TaskCard({ title, stats, datasets, taskType }: TaskCardProps) {
  return (
    <div className="bg-white rounded-2xl p-5 border-3 border-blue-400 flex flex-col h-full shadow-md">
      <h3 className="text-3xl text-gray-900 mb-3 text-center" style={{ fontWeight: 700 }}>
        {title}
      </h3>

      <div className="mb-4 flex-grow flex items-center justify-center">
        <div className="w-full">
          {taskType === "classification" && <ClassificationVisual />}
          {taskType === "retrieval"      && <RetrievalVisual />}
          {taskType === "reranking"      && <RerankingVisual />}
          {taskType === "pair"           && <PairVisual />}
          {taskType === "sts"            && <STSVisual />}
        </div>
      </div>

      <div className="mb-4">
        <div className="flex items-center justify-center gap-3 text-lg text-gray-600">
          <div className="flex items-center gap-1.5">
            <Languages className="w-4 h-4 text-blue-600" />
            <span style={{ fontWeight: 600 }}>{stats.languages}</span>
          </div>
          <span className="text-gray-400">•</span>
          <div className="flex items-center gap-1.5">
            <ListChecks className="w-4 h-4 text-blue-600" />
            <span style={{ fontWeight: 600 }}>{stats.tasks}</span>
          </div>
          <span className="text-gray-400">•</span>
          <div className="flex items-center gap-1.5">
            <Globe className="w-4 h-4 text-blue-600" />
            <span style={{ fontWeight: 600 }}>{stats.domains}</span>
          </div>
        </div>
      </div>

      <div>
        <div className="text-[13px] text-gray-500 mb-2 uppercase tracking-wide" style={{ fontWeight: 600 }}>
          Example Datasets
        </div>
        <div className="flex flex-wrap gap-1.5">
          {datasets.map((d, i) => (
            <div key={i} className="bg-blue-50 text-gray-800 px-2.5 py-1.5 rounded-md text-[12.5px] border border-blue-200" style={{ fontWeight: 500 }}>
              {d}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Dialog bubble primitives ──────────────────────────────────────────────────

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="bg-blue-500 text-white rounded-2xl rounded-br-sm px-3 py-2 text-sm max-w-[85%]" style={{ fontWeight: 500 }}>
        {text}
      </div>
    </div>
  );
}

function BotBubble({ text, highlight }: { text: string; highlight?: boolean }) {
  return (
    <div className="flex justify-start">
      <div
        className={`rounded-2xl rounded-bl-sm px-3 py-2 text-sm max-w-[85%] border ${
          highlight
            ? "bg-green-50 border-green-400 text-green-900"
            : "bg-gray-100 border-gray-200 text-gray-700"
        }`}
        style={{ fontWeight: 500 }}
      >
        {text}
      </div>
    </div>
  );
}

// ── Per-type visuals ──────────────────────────────────────────────────────────

function ClassificationVisual() {
  return (
    <div className="w-full space-y-2 px-2">
      <UserBubble text="I need to book a flight to London" />
      <BotBubble text="Sure, let me help you with that." />
      <div className="flex items-center justify-between bg-green-50 rounded-xl px-3 py-2 border-2 border-green-400 mt-1">
        <div className="text-base text-green-800" style={{ fontWeight: 700 }}>
          Intent: Flight Booking
        </div>
        <div className="text-base text-green-700" style={{ fontWeight: 700 }}>
          97%
        </div>
      </div>
      <div className="flex items-center justify-between bg-gray-50 rounded-xl px-3 py-1.5 border border-gray-200">
        <div className="text-sm text-gray-500" style={{ fontWeight: 500 }}>
          Other intents
        </div>
        <div className="text-sm text-gray-400">3%</div>
      </div>
    </div>
  );
}

function RetrievalVisual() {
  return (
    <div className="w-full space-y-2 px-2">
      <UserBubble text="What was discussed about pricing?" />
      <div className="space-y-1.5 mt-1">
        <div className="flex items-center gap-2 bg-green-50 rounded-xl px-3 py-2 border-2 border-green-400">
          <div className="flex-1 text-sm text-gray-800 truncate" style={{ fontWeight: 500 }}>
            "…the pricing plan starts at $49/mo…"
          </div>
          <div className="text-sm text-green-700 flex-shrink-0" style={{ fontWeight: 700 }}>
            0.91
          </div>
        </div>
        <div className="flex items-center gap-2 bg-gray-50 rounded-xl px-3 py-1.5 border border-gray-200">
          <div className="flex-1 text-sm text-gray-500 truncate">
            "…we offer enterprise deals…"
          </div>
          <div className="text-sm text-gray-400 flex-shrink-0">0.63</div>
        </div>
        <div className="flex items-center gap-2 bg-gray-50 rounded-xl px-3 py-1.5 border border-gray-200">
          <div className="flex-1 text-sm text-gray-500 truncate">
            "…our refund policy allows…"
          </div>
          <div className="text-sm text-gray-400 flex-shrink-0">0.41</div>
        </div>
      </div>
    </div>
  );
}

function RerankingVisual() {
  return (
    <div className="w-full space-y-2 px-2">
      <UserBubble text="How do I reset my password?" />
      <div className="space-y-1.5 mt-1">
        <div className="bg-green-50 rounded-xl px-3 py-2 border-2 border-green-400 flex items-center justify-between">
          <div className="text-sm text-gray-800" style={{ fontWeight: 700 }}>
            1. "Click Forgot Password…"
          </div>
          <div className="text-sm text-green-700" style={{ fontWeight: 700 }}>
            0.96
          </div>
        </div>
        <div className="bg-gray-50 rounded-xl px-3 py-1.5 border border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-600">2. "Contact support team…"</div>
          <div className="text-sm text-gray-400">0.72</div>
        </div>
        <div className="bg-gray-50 rounded-xl px-3 py-1.5 border border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-600">3. "Visit account settings…"</div>
          <div className="text-sm text-gray-400">0.58</div>
        </div>
      </div>
    </div>
  );
}

function PairVisual() {
  return (
    <div className="w-full space-y-3 px-2">
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-blue-50 rounded-xl p-3 border-2 border-blue-300">
          <div className="text-xs text-blue-700 mb-2 text-center" style={{ fontWeight: 700 }}>
            Query A
          </div>
          <div className="text-xs text-blue-900 text-center" style={{ fontWeight: 500 }}>
            "What's the weather like?"
          </div>
        </div>
        <div className="bg-blue-50 rounded-xl p-3 border-2 border-blue-300">
          <div className="text-xs text-blue-700 mb-2 text-center" style={{ fontWeight: 700 }}>
            Query B
          </div>
          <div className="text-xs text-blue-900 text-center" style={{ fontWeight: 500 }}>
            "Is it going to rain today?"
          </div>
        </div>
      </div>
      <div className="bg-green-500 rounded-xl px-3 py-2.5 border-2 border-green-600 text-center">
        <div className="text-sm text-white" style={{ fontWeight: 700 }}>
          Duplicate Question ✓
        </div>
      </div>
    </div>
  );
}

function STSVisual() {
  return (
    <div className="w-full space-y-2.5 px-2">
      <BotBubble text="Can you book a flight for me?" />
      <BotBubble text="I need to reserve a plane ticket." />
      <div className="mt-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-gray-500">Similarity</span>
          <span className="text-sm text-blue-700" style={{ fontWeight: 700 }}>0.91</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-3 border border-gray-200">
          <div
            className="bg-gradient-to-r from-blue-400 to-blue-600 h-3 rounded-full"
            style={{ width: "91%" }}
          />
        </div>
      </div>
    </div>
  );
}
