"use client";

import Link from "next/link";

const SAMPLES = [
  { seg: "/samples/sample_1.png", depth: "/samples/depth_1.png", gen: "/samples/generated_1.png" },
  { seg: "/samples/sample_2.png", depth: "/samples/depth_2.png", gen: "/samples/generated_2.png" },
  { seg: "/samples/sample_3.png", depth: "/samples/depth_3.png", gen: "/samples/generated_3.png" },
  { seg: "/samples/sample_4.png", depth: "/samples/depth_4.png", gen: "/samples/generated_4.png" },
];

const TEAM = [
  { name: "Harish Babu", role: "2nd Year BTech CSE" },
  { name: "Akshat Sinha", role: "2nd Year BTech EE" },
  { name: "Tejal Goel", role: "2nd Year BTech CSE" },
  { name: "Siya Patil", role: "2nd Year BTech AIDS" },
  { name: "Pranav Kuppa", role: "2nd Year BTech EE" },
];

const MENTORS = [
  { name: "Yug Dalwadi", role: "3rd Year BTech AIDS" },
  { name: "Prisha Shah", role: "3rd Year BTech ES" },
];

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="relative flex min-h-screen flex-col items-center justify-center px-8 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(255,255,255,0.04)_0%,transparent_70%)]" />

        <div className="relative z-10 max-w-4xl text-center">
          <div className="animate-fade-in-up mb-8 inline-flex items-center gap-2.5 rounded-full border border-white/10 bg-white/5 px-5 py-2 text-base text-gray-400">
            <span className="h-2 w-2 rounded-full bg-green-400" />
            Built at RAID, IIT Jodhpur
          </div>

          <h1 className="animate-fade-in-up delay-100 mb-8 text-8xl font-bold tracking-tight gradient-text">
            ViSyn
          </h1>

          <p className="animate-fade-in-up delay-200 mb-6 text-2xl text-gray-400">
            Transform segmentation maps into photorealistic landscapes
          </p>

          <p className="animate-fade-in-up delay-300 mx-auto mb-12 max-w-2xl text-lg leading-relaxed text-gray-500">
            ViSyn uses a two-stage generative adversarial network to synthesize
            realistic landscape images. Draw or upload a segmentation map, and the
            model predicts depth information, then generates a full-color landscape.
          </p>

          <div className="animate-fade-in-up delay-400 flex items-center justify-center gap-5">
            <Link
              href="/start"
              className="rounded-xl bg-white px-10 py-4 text-lg font-semibold text-black transition-all hover:bg-gray-200 hover:shadow-lg hover:shadow-white/10"
            >
              Try Now
            </Link>
            <a
              href="https://github.com/harishbabu2007/ViSyn"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-white/15 px-10 py-4 text-lg font-medium text-gray-300 transition-all hover:border-white/30 hover:text-white"
            >
              GitHub
            </a>
          </div>
        </div>

        <div className="animate-fade-in delay-700 absolute bottom-10 text-gray-600">
          <svg className="h-6 w-6 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </section>

      {/* How It Works */}
      <section className="px-8 py-28">
        <div className="mx-auto max-w-6xl">
          <h2 className="animate-fade-in-up mb-5 text-center text-4xl font-bold gradient-text">How It Works</h2>
          <p className="animate-fade-in-up delay-100 mb-16 text-center text-lg text-gray-500">
            A two-stage pipeline that adds geometric structure to semantic understanding
          </p>

          <div className="grid gap-8 md:grid-cols-3">
            <div className="animate-fade-in-up delay-200 glass rounded-2xl p-8">
              <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-white/10 text-lg font-bold text-white">1</div>
              <h3 className="mb-3 text-xl font-semibold text-white">Draw or Upload</h3>
              <p className="text-base leading-relaxed text-gray-500">
                Create a segmentation map on the 256x256 canvas or upload an existing one. Different regions represent semantic classes like sky, terrain, water.
              </p>
            </div>
            <div className="animate-fade-in-up delay-300 glass rounded-2xl p-8">
              <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-white/10 text-lg font-bold text-white">2</div>
              <h3 className="mb-3 text-xl font-semibold text-white">Depth Prediction</h3>
              <p className="text-base leading-relaxed text-gray-500">
                The S2D generator predicts a depth map from your segmentation, adding 3D spatial structure — perspective, scale, and terrain layout.
              </p>
            </div>
            <div className="animate-fade-in-up delay-400 glass rounded-2xl p-8">
              <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-white/10 text-lg font-bold text-white">3</div>
              <h3 className="mb-3 text-xl font-semibold text-white">Image Synthesis</h3>
              <p className="text-base leading-relaxed text-gray-500">
                The SD2I generator combines both signals to produce a photorealistic landscape image with consistent geometry and rich detail.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Sample Results */}
      <section className="px-8 py-28">
        <div className="mx-auto max-w-6xl">
          <h2 className="animate-fade-in-up mb-5 text-center text-4xl font-bold gradient-text">Sample Results</h2>
          <p className="animate-fade-in-up delay-100 mb-16 text-center text-lg text-gray-500">
            Segmentation maps in, photorealistic landscapes out
          </p>

          <div className="grid gap-10 sm:grid-cols-2">
            {SAMPLES.map((s, i) => (
              <div key={i} className={`animate-scale-in delay-${(i + 2) * 100} sample-card glass rounded-2xl overflow-hidden p-4`}>
                <div className="grid grid-cols-3 gap-3">
                  <div className="flex flex-col items-center gap-2">
                    <img src={s.seg} alt="Segmentation" className="aspect-square w-full rounded-lg object-cover" />
                    <p className="text-sm text-gray-600">Input</p>
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <img src={s.depth} alt="Depth" className="aspect-square w-full rounded-lg object-cover" />
                    <p className="text-sm text-gray-600">Depth</p>
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <img src={s.gen} alt="Generated" className="aspect-square w-full rounded-lg object-cover" />
                    <p className="text-sm text-gray-600">Output</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="px-8 py-28">
        <div className="mx-auto max-w-5xl">
          <h2 className="animate-fade-in-up mb-5 text-center text-4xl font-bold gradient-text">Team</h2>
          <p className="animate-fade-in-up delay-100 mb-16 text-center text-lg text-gray-500">
            Engineering team and mentors behind ViSyn
          </p>

          {/* Mentors */}
          <div className="mb-14">
            <h3 className="animate-fade-in-up delay-200 mb-8 text-center text-sm font-semibold uppercase tracking-widest text-gray-500">
              Special Thanks to Our Mentors
            </h3>
            <div className="flex justify-center gap-8">
              {MENTORS.map((m, i) => (
                <div key={i} className={`animate-fade-in-up delay-${(i + 3) * 100} team-card glass rounded-2xl px-10 py-6 text-center`}>
                  <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-white/10 text-base font-semibold text-white mx-auto">
                    {m.name.split(" ").map(n => n[0]).join("")}
                  </div>
                  <p className="text-lg font-medium text-white">{m.name}</p>
                  <p className="mt-1.5 text-sm text-gray-500">{m.role}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Engineers */}
          <div>
            <h3 className="animate-fade-in-up delay-300 mb-8 text-center text-sm font-semibold uppercase tracking-widest text-gray-500">
              Engineering Team
            </h3>
            <div className="flex flex-wrap justify-center gap-6">
              {TEAM.map((t, i) => (
                <div key={i} className={`animate-fade-in-up delay-${(i + 4) * 100} team-card glass rounded-2xl px-8 py-5 text-center`}>
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-white/10 text-sm font-semibold text-white mx-auto">
                    {t.name.split(" ").map(n => n[0]).join("")}
                  </div>
                  <p className="text-base font-medium text-white">{t.name}</p>
                  <p className="mt-1.5 text-sm text-gray-500">{t.role}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* RAID + Footer */}
      <section className="px-8 py-24">
        <div className="mx-auto max-w-4xl">
          <div className="animate-fade-in-up glass rounded-3xl p-12 text-center">
            <a href="https://raid.iitj.ac.in/" target="_blank" rel="noopener noreferrer" className="inline-block">
              <img
                src="/RAID_logo.jpeg"
                alt="RAID IIT Jodhpur"
                className="mx-auto mb-6 h-24 w-24 rounded-2xl object-contain"
              />
            </a>
            <p className="mb-2 text-lg text-gray-400">
              A project by{" "}
              <a
                href="https://raid.iitj.ac.in/"
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-white underline underline-offset-4 decoration-white/30 transition-colors hover:decoration-white"
              >
                RAID Society
              </a>
              {" "}at IIT Jodhpur
            </p>
            <p className="mb-8 text-sm text-gray-600">
              Research and Development in AI and Data Science
            </p>
            <div className="flex items-center justify-center gap-5">
              <a
                href="https://raid.iitj.ac.in/"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-xl border border-white/15 px-6 py-3 text-base font-medium text-gray-300 transition-all hover:border-white/30 hover:text-white"
              >
                RAID Website
              </a>
              <a
                href="https://github.com/harishbabu2007/ViSyn"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-xl border border-white/15 px-6 py-3 text-base font-medium text-gray-300 transition-all hover:border-white/30 hover:text-white"
              >
                GitHub Repo
              </a>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-white/5 px-8 py-10 text-center text-sm text-gray-700">
        ViSyn &mdash; StyleGAN-inspired conditional landscape synthesis
      </footer>
    </main>
  );
}
