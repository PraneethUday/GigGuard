import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

export async function PATCH(req: NextRequest) {
  try {
    const { claimId, ...body } = await req.json();
    if (!claimId) return NextResponse.json({ error: "claimId required" }, { status: 400 });

    const res = await fetch(`${BACKEND}/api/claims/${claimId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ error: "Backend unavailable." }, { status: 502 });
  }
}
