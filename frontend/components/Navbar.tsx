"use client";

import { useState } from "react";
import { TrendingUp, Clock, Activity } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="flex items-center justify-between p-4 bg-gray-900 border-b border-gray-800 text-white shadow-md">
      <div className="flex items-center space-x-2">
        <TrendingUp className="text-green-500 w-6 h-6" />
        <span className="text-xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
          Green Dragon Core
        </span>
      </div>
    </nav>
  );
}
