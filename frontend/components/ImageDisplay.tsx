"use client";

import { useState } from "react";
import { getFileUrl } from "@/lib/api";

interface ImageDisplayProps {
  label: string;
  path: string;
}

export default function ImageDisplay({ label, path }: ImageDisplayProps) {
  const [loaded, setLoaded] = useState(false);

  const handleDownload = async () => {
    const url = getFileUrl(path);
    const res = await fetch(url);
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = path.split("/").pop() || "image.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  };

  return (
    <div className="flex flex-col items-center gap-3">
      <span className="text-sm text-gray-500">{label}</span>
      <div className="group relative rounded-xl border border-gray-700 bg-gray-800 p-1.5">
        {!loaded && (
          <div className="flex h-[320px] w-[320px] items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-600 border-t-white" />
          </div>
        )}
        <img
          src={getFileUrl(path)}
          alt={label}
          width={320}
          height={320}
          onLoad={() => setLoaded(true)}
          className={loaded ? "block rounded-lg" : "hidden"}
        />
        {loaded && (
          <button
            onClick={handleDownload}
            className="absolute top-3 right-3 flex h-9 w-9 items-center justify-center rounded-lg bg-black/60 text-gray-300 opacity-0 transition-opacity hover:bg-black/80 hover:text-white group-hover:opacity-100"
            title="Download"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v12m0 0l-4-4m4 4l4-4M4 18h16" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
