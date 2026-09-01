"use client";

import { useRef, useState, useCallback, useEffect } from "react";

interface CanvasProps {
  onCapture: (file: File) => void;
}

export default function Canvas({ onCapture }: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokeSize, setStrokeSize] = useState(8);
  const [greyIntensity, setGreyIntensity] = useState(128);
  const [hasContent, setHasContent] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const getPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const startDraw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const ctx = canvasRef.current!.getContext("2d")!;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const ctx = canvasRef.current!.getContext("2d")!;
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    const hex = greyIntensity.toString(16).padStart(2, "0");
    ctx.strokeStyle = `#${hex}${hex}${hex}`;
    ctx.lineWidth = strokeSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
    setHasContent(true);
  };

  const endDraw = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasContent(false);
  };

  const captureCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "segment_map.png", {
          type: "image/png",
        });
        onCapture(file);
      }
    }, "image/png");
  }, [onCapture]);

  const previewHex = greyIntensity.toString(16).padStart(2, "0");

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="rounded-xl border border-gray-700 p-1.5">
        <canvas
          ref={canvasRef}
          width={256}
          height={256}
          className="cursor-crosshair bg-black rounded-lg"
          style={{ width: 512, height: 512, imageRendering: "pixelated" }}
          onMouseDown={startDraw}
          onMouseMove={draw}
          onMouseUp={endDraw}
          onMouseLeave={endDraw}
        />
      </div>

      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <label className="text-base text-gray-400">Stroke:</label>
            <input
              type="range"
              min="1"
              max="20"
              value={strokeSize}
              onChange={(e) => setStrokeSize(Number(e.target.value))}
              className="w-32 accent-white"
            />
            <span className="w-8 text-center text-sm text-gray-500">
              {strokeSize}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-base text-gray-400">Intensity:</label>
            <input
              type="range"
              min="10"
              max="255"
              value={greyIntensity}
              onChange={(e) => setGreyIntensity(Number(e.target.value))}
              className="w-32 accent-white"
            />
            <div
              className="h-5 w-5 rounded border border-gray-600"
              style={{ backgroundColor: `#${previewHex}${previewHex}${previewHex}` }}
            />
          </div>

          <button
            onClick={clearCanvas}
            className="rounded-lg border border-gray-600 px-4 py-2 text-base text-gray-400 transition-colors hover:border-gray-400 hover:text-white"
          >
            Clear
          </button>
        </div>
      </div>

      <button
        onClick={captureCanvas}
        disabled={!hasContent}
        className="rounded-xl bg-white px-8 py-3 text-base font-semibold text-gray-900 transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-30"
      >
        Generate Depth Map
      </button>
    </div>
  );
}
