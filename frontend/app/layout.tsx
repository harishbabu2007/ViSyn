import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ViSyn",
  description: "Transform segmentation maps into photorealistic landscapes",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-900 text-gray-50 antialiased">
        {children}
      </body>
    </html>
  );
}
