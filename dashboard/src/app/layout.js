import "./globals.css";

export const metadata = {
  title: "IntelliCrash — Dual-Phase Vehicular Telematics & Crash Anticipation Console",
  description: "Physics-Informed Deep Learning & Edge AI Telematics Console (CPG-257, Thapar Institute of Engineering and Technology)",
  icons: {
    icon: "/assets/logo.jpg",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#f8fafc] text-slate-900 selection:bg-sky-500/20">
        {children}
      </body>
    </html>
  );
}
