"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import StatusBadge from "@/components/StatusBadge";
import ImageDisplay from "@/components/ImageDisplay";
import { getJobStatus, generateColor, JobStatus } from "@/lib/api";

export default function JobPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.jobId as string;

  const [job, setJob] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [colorLoading, setColorLoading] = useState(false);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getJobStatus(jobId);
      setJob(status);
      setLoading(false);

      if (status.status === "processing") {
        pollRef.current = setTimeout(fetchStatus, 10000);
      }
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchStatus();
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [fetchStatus]);

  const handleDownloadAll = async () => {
    if (!job?.images) return;
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    for (const [key, url] of Object.entries(job.images)) {
      const res = await fetch(`${API_BASE}${url}`);
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = url.split("/").pop() || `${key}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
      await new Promise((r) => setTimeout(r, 300));
    }
  };

  const handleGenerateColor = async () => {
    if (!job?.images?.segment_map || !job.images.depth_map) return;

    setColorLoading(true);
    setError(null);

    try {
      const API_BASE =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const segRes = await fetch(`${API_BASE}${job.images.segment_map}`);
      const depthRes = await fetch(`${API_BASE}${job.images.depth_map}`);

      const segBlob = await segRes.blob();
      const depthBlob = await depthRes.blob();

      const segFile = new File([segBlob], "segment.png", { type: "image/png" });
      const depthFile = new File([depthBlob], "depth.png", {
        type: "image/png",
      });

      const { job_id } = await generateColor(segFile, depthFile);
      router.push(`/job/${job_id}`);
    } catch (err: any) {
      setError(err.message || "Failed to generate color image");
      setColorLoading(false);
    }
  };

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <div className="h-10 w-10 animate-spin rounded-full border-2 border-gray-600 border-t-white" />
      </main>
    );
  }

  if (!job) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-5">
        <p className="text-xl text-gray-400">Job not found</p>
        <button
          onClick={() => router.push("/start")}
          className="text-base text-gray-500 hover:text-white"
        >
          Go back
        </button>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center px-8 py-16">
      <div className="mb-10 flex items-center gap-5">
        <h1 className="text-3xl font-bold text-white">
          {job.job_type === "depth" ? "Depth Map Generation" : "Color Image Generation"}
        </h1>
        <StatusBadge status={job.status} />
      </div>

      {job.status === "processing" && (
        <div className="flex flex-col items-center gap-5">
          <div className="animate-pulse-slow text-xl text-gray-400">
            {job.job_type === "depth"
              ? "Generating your depth map..."
              : "Generating your colored image..."}
          </div>
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-gray-600 border-t-white" />
          <p className="text-sm text-gray-600">
            This page refreshes automatically every 10 seconds
          </p>
        </div>
      )}

      {job.status === "failed" && (
        <div className="flex flex-col items-center gap-5">
          <p className="text-base text-red-400">
            {job.error || "Something went wrong"}
          </p>
          <button
            onClick={() => router.push("/start")}
            className="rounded-xl border border-gray-700 px-6 py-3 text-base text-gray-400 hover:border-gray-500 hover:text-white"
          >
            Try Again
          </button>
        </div>
      )}

      {job.status === "completed" && (
        <div className="flex flex-col items-center gap-10">
          <div className="flex flex-wrap justify-center gap-8">
            {job.images?.segment_map && (
              <ImageDisplay
                label="Segmentation Map"
                path={job.images.segment_map}
              />
            )}
            {job.images?.depth_map && (
              <ImageDisplay label="Depth Map" path={job.images.depth_map} />
            )}
            {job.images?.generated_image && (
              <ImageDisplay
                label="Generated Image"
                path={job.images.generated_image}
              />
            )}
          </div>

          {job.job_type === "depth" && !job.images?.generated_image && (
            <button
              onClick={handleGenerateColor}
              disabled={colorLoading}
              className="rounded-xl bg-white px-8 py-3 text-base font-semibold text-gray-900 transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {colorLoading ? "Submitting..." : "Generate Colored Image"}
            </button>
          )}

          {job.images?.generated_image && (
            <button
              onClick={handleDownloadAll}
              className="flex items-center gap-2.5 rounded-xl border border-white/15 px-8 py-3 text-base font-medium text-gray-300 transition-all hover:border-white/30 hover:text-white"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v12m0 0l-4-4m4 4l4-4M4 18h16" />
              </svg>
              Download All Images
            </button>
          )}

          <button
            onClick={() => router.push("/start")}
            className="text-base text-gray-500 hover:text-white"
          >
            Start Over
          </button>
        </div>
      )}

      {error && (
        <p className="mt-6 rounded-xl bg-red-900/30 px-5 py-3 text-base text-red-400">
          {error}
        </p>
      )}
    </main>
  );
}
