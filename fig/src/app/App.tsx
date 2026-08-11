import { TaskCard } from "./components/TaskCard";
import { Languages, ListChecks, Globe } from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen bg-white p-8 flex items-center justify-center">
      <div className="w-full max-w-[1850px]">
        <div id="dialogmteb-figure" className="bg-white">
          <div className="grid grid-cols-2 grid-rows-2 gap-4 items-stretch">

            {/* Top-left — header */}
            <div className="bg-gradient-to-br from-blue-50 via-white to-indigo-50 border-4 border-blue-400 rounded-3xl p-8 shadow-lg h-full">
              <div className="h-full max-w-[720px] mx-auto flex flex-col justify-center pt-3">
                <div className="text-center mb-8">
                  <h1
                    className="text-6xl text-gray-900 mb-3"
                    style={{ fontWeight: 900, letterSpacing: "-0.02em" }}
                  >
                    DialogMTEB(v1)
                  </h1>
                  <div className="text-xl text-gray-600" style={{ fontWeight: 500, letterSpacing: "0.01em" }}>
                    <span style={{ fontWeight: 700 }}>Dialog</span>ue{" "}
                    <span style={{ fontWeight: 700 }}>M</span>assive{" "}
                    <span style={{ fontWeight: 700 }}>T</span>ext{" "}
                    <span style={{ fontWeight: 700 }}>E</span>mbedding{" "}
                    <span style={{ fontWeight: 700 }}>B</span>enchmark
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-6 mt-3">
                  {[
                    { icon: Languages, value: "51", label: "Languages" },
                    { icon: ListChecks, value: "29", label: "Tasks" },
                    { icon: Globe,      value: "8",  label: "Domains" },
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

            {/* Top-right */}
            <TaskCard
              title="Classification"
              stats={{ languages: "50", tasks: "17", domains: "6" }}
              datasets={[
                "MTOP Intent",
                "Massive Intent",
                "Banking77",
                "MultiWoz21",
              ]}
              taskType="classification"
            />

            {/* Bottom-left */}
            <TaskCard
              title="Retrieval"
              stats={{ languages: "3", tasks: "9", domains: "5" }}
              datasets={[
                "Statcan Dialogue",
                "FaithDial",
                "Clarq",
                "TopiOCQA",
              ]}
              taskType="retrieval"
            />

            {/* Bottom-right */}
            <TaskCard
              title="Pair Classification"
              stats={{ languages: "3", tasks: "2", domains: "0" }}
              datasets={["ClarQA", "QRECC"]}
              taskType="pair"
            />

          </div>
        </div>
      </div>
    </div>
  );
}
