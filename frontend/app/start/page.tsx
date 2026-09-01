"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Canvas from "@/components/Canvas";
import { generateDepth } from "@/lib/api";

export default function StartPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"draw" | "upload">("draw");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleCanvasCapture = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const { job_id } = await generateDepth(file);
      router.push(`/job/${job_id}`);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
      setLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const { job_id } = await generateDepth(selectedFile);
      router.push(`/job/${job_id}`);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-8 py-16">
      <h1 className="mb-3 text-4xl font-bold text-white">Create Your Map</h1>
      <p className="mb-10 text-lg text-gray-500">
        Draw a segmentation map or upload an existing one
      </p>

      <div className="mb-8 flex gap-5">
        <button
          onClick={() => setMode("draw")}
          className={`rounded-xl px-6 py-3 text-base font-medium transition-colors ${
            mode === "draw"
              ? "bg-white text-gray-900"
              : "border border-gray-700 text-gray-400 hover:border-gray-500"
          }`}
        >
          Draw
        </button>
        <button
          onClick={() => setMode("upload")}
          className={`rounded-xl px-6 py-3 text-base font-medium transition-colors ${
            mode === "upload"
              ? "bg-white text-gray-900"
              : "border border-gray-700 text-gray-400 hover:border-gray-500"
          }`}
        >
          Upload File
        </button>
      </div>

      {mode === "draw" ? (
        <Canvas onCapture={handleCanvasCapture} />
      ) : (
        <div className="flex flex-col items-center gap-5">
          <div
            onClick={() => fileInputRef.current?.click()}
            className="flex h-[512px] w-[512px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-700 transition-colors hover:border-gray-500"
          >
            {selectedFile ? (
              <img
                src={URL.createObjectURL(selectedFile)}
                alt="Upload preview"
                className="max-h-[480px] object-contain"
              />
            ) : (
              <>
                <svg
                  className="mb-3 h-14 w-14 text-gray-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                <span className="text-base text-gray-500">
                  Click to upload PNG/JPEG
                </span>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg"
            onChange={handleFileUpload}
            className="hidden"
          />

          <button
            onClick={handleUploadSubmit}
            disabled={!selectedFile}
            className="rounded-xl bg-white px-8 py-3 text-base font-semibold text-gray-900 transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-30"
          >
            Generate Depth Map
          </button>
        </div>
      )}

      {error && (
        <p className="mt-6 rounded-xl bg-red-900/30 px-5 py-3 text-base text-red-400">
          {error}
        </p>
      )}

      {loading && (
        <div className="mt-8 flex items-center gap-3 text-base text-gray-400">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-600 border-t-white" />
          Submitting job...
        </div>
      )}
    </main>
  );
}
