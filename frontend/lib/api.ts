const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function generateDepth(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("segment_map", file);

  const res = await fetch(`${API_BASE}/api/generate-depth`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to submit depth generation job");
  }

  return res.json();
}

export async function generateColor(
  segFile: File,
  depthFile: File
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("segment_map", segFile);
  form.append("depth_map", depthFile);

  const res = await fetch(`${API_BASE}/api/generate-color`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to submit color generation job");
  }

  return res.json();
}

export interface JobStatus {
  job_id: string;
  status: "processing" | "completed" | "failed";
  job_type: "depth" | "color";
  error?: string;
  images?: {
    segment_map?: string;
    depth_map?: string;
    generated_image?: string;
  };
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/job/${jobId}`);

  if (!res.ok) {
    throw new Error("Failed to fetch job status");
  }

  return res.json();
}

export function getFileUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
