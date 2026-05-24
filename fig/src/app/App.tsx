import { TaskCard } from "./components/TaskCard";
import { Languages, ListChecks, Globe, Layers } from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen bg-white p-8 flex items-center justify-center">
      <div className="w-full max-w-[1850px]">
        <div id="dialogmteb-figure" className="bg-white">
          <div className="grid grid-cols-[340px_1fr_340px] grid-rows-[auto_auto] gap-4 items-stretch">

            {/* Row 1 left */}
            <TaskCard
              title="Classification"
              stats={{ languages: "50", tasks: "14", domains: "7" }}
              datasets={[
                "ATIS Intent",
                "Banking77",
                "DailyDialog Emotion",
                "MASSIVE Intent",
              ]}
              taskType="classification"
            />

            {/* Row 1 centre — header */}
            <div className="bg-gradient-to-br from-blue-50 via-white to-indigo-50 border-4 border-blue-400 rounded-3xl p-8 shadow-lg h-full">
              <div className="h-full max-w-[1120px] mx-auto flex flex-col justify-center pt-3">
                <div className="text-center mb-8">
                  <h1
                    className="text-6xl text-gray-900 mb-3"
                    style={{ fontWeight: 900, letterSpacing: "-0.02em" }}
                  >
                    DialogMTEB
                  </h1>
                  <div className="text-xl text-gray-600" style={{ fontWeight: 500, letterSpacing: "0.01em" }}>
                    <span style={{ fontWeight: 700 }}>Dialog</span>ue{" "}
                    <span style={{ fontWeight: 700 }}>M</span>assive{" "}
                    <span style={{ fontWeight: 700 }}>T</span>ext{" "}
                    <span style={{ fontWeight: 700 }}>E</span>mbedding{" "}
                    <span style={{ fontWeight: 700 }}>B</span>enchmark
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-6 mt-3">
                  {[
                    { icon: Languages, value: "51", label: "Languages" },
                    { icon: ListChecks, value: "25", label: "Tasks" },
                    { icon: Globe,      value: "9",  label: "Domains" },
                    { icon: Layers,     value: "50+", label: "Models" },
                  ].map(({ icon: Icon, value, label }) => (
                    <div key={label} className="text-center">
                      <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-3 shadow-lg">
                        <Icon className="w-11 h-11 text-white" strokeWidth={2.5} />
                      </div>
                      <div className="text-3xl text-gray-900 mb-1.5" style={{ fontWeight: 800 }}>
                        {value}
                      </div>
                      <div className="text-base text-gray-600" style={{ fontWeight: 600 }}>
                        {label}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Row 1 right */}
            <TaskCard
              title="Retrieval"
              stats={{ languages: "7", tasks: "8", domains: "6" }}
              datasets={[
                "Canard",
                "FaithDial",
                "TopiOCQA",
                "iKAT 2023",
              ]}
              taskType="retrieval"
            />

            {/* Row 2 — 3 equal cards spanning all columns */}
            <div className="col-span-3 grid grid-cols-3 gap-4">
              <TaskCard
                title="STS"
                stats={{ languages: "1", tasks: "1", domains: "2" }}
                datasets={["ATEC"]}
                taskType="sts"
              />
              <TaskCard
                title="Pair Classification"
                stats={{ languages: "1", tasks: "1", domains: "1" }}
                datasets={["ClarQA"]}
                taskType="pair"
              />
              <TaskCard
                title="Reranking"
                stats={{ languages: "1", tasks: "1", domains: "3" }}
                datasets={["WebLINX Reranking"]}
                taskType="reranking"
              />
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
