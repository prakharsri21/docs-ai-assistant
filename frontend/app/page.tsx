"use client";

import { FormEvent, useState } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [message, setMessage] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    // Authentication will be connected to FastAPI on Day 5.
    setMessage("Login API will be connected after we build authentication.");
  }

  return (
    <main className="min-h-screen w-full bg-[#0f131d] text-[#dfe2f1]">
      <div className="mx-auto flex min-h-screen w-full max-w-2xl flex-col px-4 pb-8 pt-2 sm:px-6">
        {/* Cohort */}
        <div className="mb-6 mt-2 flex justify-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-[#262a35]/70 px-3 py-1 backdrop-blur-md">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#4edea3]" />

            <span
              className="text-[11px] font-medium uppercase tracking-wider text-[#adc6ff]"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              INSTITUTIONAL ACCESS • FALL 2025 COHORT
            </span>
          </div>
        </div>

        {/* Brand */}
        <div className="mb-6 flex flex-col items-center text-center">
          <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-xl bg-[#1c1f2a] p-2 shadow-lg shadow-black/40">
            <div className="flex h-full w-full items-center justify-center rounded-lg bg-[#313540]">
              <span className="material-symbols-outlined text-3xl text-[#adc6ff]">
                school
              </span>
            </div>
          </div>

          <h1
            className="text-3xl font-bold tracking-tight text-[#dfe2f1]"
            style={{ fontFamily: "Space Grotesk, sans-serif" }}
          >
            Course AI
          </h1>

          <p
            className="mt-1 max-w-xs text-sm leading-snug text-[#c2c6d6]"
            style={{ fontFamily: "Geist, sans-serif" }}
          >
            Academic Intelligence Grounded in Your 12 Course Modules
          </p>
        </div>

        {/* Trust badges */}
        <div className="mb-6 grid grid-cols-1 gap-2">
          <div className="flex items-center gap-2 rounded-lg bg-[#171b26] px-4 py-2.5">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#4edea3]/10">
              <span className="material-symbols-outlined text-[18px] text-[#4edea3]">
                verified_user
              </span>
            </div>

            <p className="text-sm text-[#c2c6d6]">
              Answers are grounded in the approved course materials
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-lg bg-[#171b26] px-4 py-2.5">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#adc6ff]/10">
              <span className="material-symbols-outlined text-[18px] text-[#adc6ff]">
                shield
              </span>
            </div>

            <p className="text-sm text-[#c2c6d6]">
              Questions outside the verified course scope will be declined
            </p>
          </div>
        </div>

        {/* Login card */}
        <div className="flex w-full flex-col gap-4 rounded-xl bg-[#171b26] p-4 shadow-xl">
          {/* Course indicator */}
          <div className="flex items-center justify-between rounded-lg bg-[#1c1f2a] px-2 py-2">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px] text-[#adc6ff]">
                school
              </span>

              <span
                className="text-[13px] font-semibold text-[#dfe2f1]"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                COURSE AI
              </span>
            </div>

            <div className="flex items-center gap-1.5 rounded-full bg-[#4edea3]/10 px-2 py-0.5">
              <span className="h-1.5 w-1.5 rounded-full bg-[#4edea3]" />

              <span className="text-[11px] font-medium text-[#4edea3]">
                Active
              </span>
            </div>
          </div>

          <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
            {/* Email */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <label
                  htmlFor="email"
                  className="text-sm font-medium text-[#dfe2f1]"
                >
                  Email
                </label>

                <span className="text-[11px] text-[#8c909f]">
                  Registered account
                </span>
              </div>

              <div className="relative flex items-center">
                <span className="material-symbols-outlined pointer-events-none absolute left-3 text-[20px] text-[#8c909f]">
                  mail
                </span>

                <input
                  id="email"
                  type="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="you@example.com"
                  className="h-11 w-full rounded-lg bg-[#0a0e18] pl-10 pr-3 text-sm text-[#f8fafc] placeholder:text-[#64748b] outline-none transition-colors focus:bg-[#262a35] focus:ring-2 focus:ring-[#3b82f6]/20"
                />
              </div>
            </div>

            {/* Password */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between">
                <label
                  htmlFor="password"
                  className="text-sm font-medium text-[#dfe2f1]"
                >
                  Password
                </label>

                <button
                  type="button"
                  onClick={() =>
                    setMessage("Password recovery will be implemented later.")
                  }
                  className="text-[11px] text-[#adc6ff] hover:underline"
                >
                  Forgot password?
                </button>
              </div>

              <div className="relative flex items-center">
                <span className="material-symbols-outlined pointer-events-none absolute left-3 text-[20px] text-[#8c909f]">
                  lock
                </span>

                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Enter your password"
                  className="h-11 w-full rounded-lg bg-[#0a0e18] pl-10 pr-11 text-sm text-[#f8fafc] placeholder:text-[#64748b] outline-none transition-colors focus:bg-[#262a35] focus:ring-2 focus:ring-[#3b82f6]/20"
                />

                <button
                  type="button"
                  aria-label={
                    showPassword ? "Hide password" : "Show password"
                  }
                  onClick={() => setShowPassword((value) => !value)}
                  className="absolute right-3 flex items-center justify-center text-[#8c909f] transition-colors hover:text-[#dfe2f1]"
                >
                  <span className="material-symbols-outlined text-[20px]">
                    {showPassword ? "visibility_off" : "visibility"}
                  </span>
                </button>
              </div>
            </div>

            {/* Remember */}
            <label className="flex cursor-pointer items-center gap-2.5">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(event) => setRememberMe(event.target.checked)}
                className="h-4 w-4 accent-[#3b82f6]"
              />

              <span className="text-sm text-[#c2c6d6]">
                Remember this device
              </span>
            </label>

            {/* Sign in */}
            <button
              type="submit"
              className="flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-[#4d8eff] font-semibold text-[#00285d] shadow-lg shadow-[#4d8eff]/20 transition-all hover:bg-[#5da0ff] active:scale-[0.99]"
            >
              <span>Sign in to Course AI</span>

              <span className="material-symbols-outlined text-[20px]">
                arrow_forward
              </span>
            </button>
          </form>

          {/* Temporary message */}
          {message && (
            <div className="rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#94a3b8]">
              {message}
            </div>
          )}
        </div>

        {/* Boundary notice */}
        <div className="mt-6 rounded-xl bg-[#1c1f2a]/60 p-4">
          <div className="flex items-start gap-2">
            <div className="mt-0.5 flex shrink-0 items-center justify-center rounded-lg bg-[#262a35] p-1.5">
              <span className="material-symbols-outlined text-[20px] text-[#4edea3]">
                verified
              </span>
            </div>

            <div className="flex flex-col gap-1">
              <h2 className="text-[15px] font-semibold text-[#dfe2f1]">
                Restricted Knowledge Boundary
              </h2>

              <p className="text-sm leading-relaxed text-[#c2c6d6]">
                The assistant will answer only from the approved course
                modules. Medical, supplement-dosing, and unrelated questions
                will be declined.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 flex flex-col items-center gap-2 text-center text-[#8c909f]">
          <div
            className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[11px]"
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            <span>
              Course Knowledge Base{" "}
              <span className="font-semibold text-[#adc6ff]">
                Modules 1–12
              </span>
            </span>

            <span className="h-1 w-1 rounded-full bg-[#8c909f]" />

            <span className="text-[#4edea3]">Verified sources</span>
          </div>

          <p className="text-[10px] leading-tight text-[#64748b]">
            Session queries are private to the registered student.
          </p>
        </div>
      </div>
    </main>
  );
}