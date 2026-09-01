interface StatusBadgeProps {
  status: "processing" | "completed" | "failed";
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const styles = {
    processing: "bg-yellow-900/50 text-yellow-300 border-yellow-700",
    completed: "bg-green-900/50 text-green-300 border-green-700",
    failed: "bg-red-900/50 text-red-300 border-red-700",
  };

  return (
    <span
      className={`inline-block rounded-full border px-3 py-1 text-xs font-medium ${styles[status]}`}
    >
      {status === "processing" && "Processing..."}
      {status === "completed" && "Completed"}
      {status === "failed" && "Failed"}
    </span>
  );
}
